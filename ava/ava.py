#!/usr/bin/env python3

"""
AVA Test Runner - Video Codec Testing Framework

This is the main entry point for the AVA video codec testing framework.
It replaces the pytest-based approach with a custom test runner that follows
a three-step process: data collection, transformation, and analysis.

Usage:
    python3 ava.py --test qp_bounds --encoder c2.qti.hevc.encoder -s <serial_number>
    python3 ava.py --test-list
    python3 ava.py --encoder hevc --input-file video.mp4
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import our modules
from .ava_quality import QualityAssessment
from .ava_report import ReportGenerator, TestResult as ReportTestResult

# Import the test runner components directly
import importlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional modules - only imported when needed
ENCAPP_AVAILABLE = False

def _import_optional_modules():
    """Import optional modules only when needed"""
    global ENCAPP_AVAILABLE
    # Import ava_common first (it handles encapp imports internally)
    from . import ava_common
    ENCAPP_AVAILABLE = True
    # ava_version is optional
    ava_version = None
    return ava_common.encapp, ava_common, ava_version


@dataclass
class TestConfig:
    """Configuration for test execution"""
    debug: int = 0
    dry_run: bool = False
    android_serials: List[str] = None
    encoder: Optional[str] = None
    test_name: Optional[str] = None
    test_list: bool = False
    input_files: List[str] = None
    output_dir: Optional[str] = None
    workdir: Optional[str] = None
    max_workers: int = 1
    use_yuv_encoding: bool = False
    enable_device_decode: bool = True
    mediastore: str = "_mediastore"
    quality_data: Optional[List[str]] = None
    quality_data_labels: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.android_serials is None:
            self.android_serials = []
        if self.input_files is None:
            self.input_files = [f"{Path(__file__).parent}/../vid/johnny.1280x720.60fps.264.mp4"]
        if self.quality_data is None:
            self.quality_data = []
        if self.quality_data_labels is None:
            self.quality_data_labels = []


@dataclass
class TestResult:
    """Result of a test execution"""
    test_name: str
    device_serial: str
    success: bool
    duration: float
    output_files: List[str]
    test_data: Dict[str, Any]
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


class TestDiscovery:
    """Discovers test files and functions following the naming convention"""
    
    def __init__(self, test_dir: str = None):
        if test_dir is None:
            # Use the package directory containing this file
            package_dir = Path(__file__).parent
            self.test_dir = package_dir
        else:
            self.test_dir = Path(test_dir)
        self.test_files = []
        self.test_functions = {}
        
    def discover_tests(self) -> Dict[str, List[callable]]:
        """Discover all test files and functions"""
        self.test_files = list(self.test_dir.rglob("test_*.py"))
        
        for test_file in self.test_files:
            try:
                # Import the test module
                module_name = test_file.stem
                spec = importlib.util.spec_from_file_location(module_name, test_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all functions starting with 'test_'
                test_functions = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (callable(attr) and 
                        attr_name.startswith('test_') and 
                        not attr_name.startswith('__')):
                        test_functions.append(attr)
                
                if test_functions:
                    self.test_functions[str(test_file)] = test_functions
                    
            except Exception as e:
                print(f"Warning: Could not load test file {test_file}: {e}")
                
        return self.test_functions
    
    def get_test_by_name(self, test_name: str) -> Optional[callable]:
        """Get a specific test function by name"""
        for file_path, functions in self.test_functions.items():
            for func in functions:
                if func.__name__ == test_name:
                    return func
        return None


class DeviceManager:
    """Manages device discovery and codec lookup"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.devices = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debug information"""
        logger = logging.getLogger("ava_device_manager")
        logger.setLevel(logging.DEBUG if self.config.debug > 0 else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def discover_devices(self) -> List[Dict[str, Any]]:
        """Discover connected devices and their capabilities"""
        self.logger.info("Starting device discovery...")
        encapp, ava_common, ava_version = _import_optional_modules()
        self.logger.info("Using encapp for device discovery")
        
        # Use ava_common for device discovery if available
        try:
            self.logger.info("Getting device serials...")
            # Get device serials using ava_common
            serials = ava_common.get_all_serials()
            if not serials:
                self.logger.error("No devices found")
                return []
            
            self.logger.info(f"Found {len(serials)} devices: {serials}")
            devices = []
            for serial in serials:
                try:
                    self.logger.info(f"Setting up device {serial}...")
                    # Ensure encapp is installed
                    self.logger.info(f"Checking encapp installation for device {serial}...")
                    ava_common.encapp_is_installed(serial, self.config.debug)
                    self.logger.info(f"Encapp installation check completed for device {serial}")
                    
                    # Get device workdir with timeout
                    self.logger.info(f"Getting workdir for device {serial}...")
                    try:
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("get_workdir timed out")
                        
                        # Set timeout for get_workdir call
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(10)  # 10 second timeout
                        
                        device_workdir = encapp.get_workdir(serial)
                        signal.alarm(0)  # Cancel timeout
                        
                        self.logger.info(f"Device {serial} workdir: {device_workdir}")
                    except TimeoutError:
                        self.logger.warning(f"get_workdir timed out for device {serial}, using default")
                        device_workdir = "/sdcard"
                    except Exception as e:
                        self.logger.warning(f"Failed to get workdir for device {serial}: {e}, using default")
                        device_workdir = "/sdcard"
                    
                    # Get codec information
                    self.logger.info(f"Getting codecs for device {serial}...")
                    codecs = ava_common.list_codecs_for_serial(serial)
                    encoders = codecs.get("encoders", [])
                    self.logger.info(f"Found {len(encoders)} encoders for device {serial}")
                    
                    # Filter encoders based on criteria
                    filtered_encoders = self._filter_encoders(encoders)
                    self.logger.info(f"Filtered to {len(filtered_encoders)} suitable encoders")
                    
                    if not filtered_encoders:
                        self.logger.warning(f"No suitable encoders found for device {serial}")
                        continue
                        
                    # If specific encoder requested, find matching one
                    if self.config.encoder:
                        matching_encoders = [enc for enc in filtered_encoders 
                                           if self.config.encoder.lower() in enc.get("name", "").lower()]
                        if not matching_encoders:
                            self.logger.warning(f"Requested encoder '{self.config.encoder}' not found on device {serial}")
                            continue
                        filtered_encoders = matching_encoders
                    
                    # Create device info for each encoder
                    for encoder_info in filtered_encoders:
                        # Use canonical_name if available, fallback to name
                        encoder_name = encoder_info.get("canonical_name", encoder_info.get("name", ""))
                        device = {
                            "serial": serial,
                            "device_workdir": device_workdir,
                            "encoder": encoder_name,
                            "encoder_info": encoder_info,
                            "codecs": codecs
                        }
                        devices.append(device)
                        self.logger.info(f"Added device {serial} with encoder {encoder_name}")
                        
                except Exception as e:
                    self.logger.warning(f"Could not setup device {serial}: {e}")
                    
            self.logger.info(f"Device discovery complete. Found {len(devices)} device/encoder combinations")
            return devices
            
        except Exception as e:
            self.logger.error(f"Failed to discover devices with ava_common: {e}")
            return self._discover_devices_with_adb()
    
    def _discover_devices_with_adb(self) -> List[Dict[str, Any]]:
        """Discover devices using ADB directly"""
        import subprocess
        import json
        
        try:
            # Get connected devices
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.error(f"ADB command failed: {result.stderr}")
                return []
            
            devices = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    serial = parts[0]
                    status = parts[1]
                    
                    if status == "unauthorized":
                        self.logger.error(f"Device {serial} is unauthorized. Please authorize ADB access on the device.")
                        continue
                    elif status != "device":
                        self.logger.warning(f"Device {serial} status: {status}")
                        continue
                    
                    # Get device info
                    device_info = self._get_device_info(serial)
                    if device_info:
                        devices.append(device_info)
            
            if not devices:
                self.logger.error("No authorized devices found. Please check device connection and authorization.")
                return []
                
            return devices
            
        except subprocess.TimeoutExpired:
            self.logger.error("ADB command timed out")
            return []
        except FileNotFoundError:
            self.logger.error("ADB not found. Please ensure Android SDK is installed and ADB is in PATH.")
            return []
        except Exception as e:
            self.logger.error(f"Failed to discover devices with ADB: {e}")
            return []
    
    def _get_device_info(self, serial: str) -> Dict[str, Any]:
        """Get device information for a given serial"""
        import subprocess
        
        try:
            # Get device properties
            result = subprocess.run(['adb', '-s', serial, 'shell', 'getprop'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None
            
            # Parse device properties
            props = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    props[key.strip()] = value.strip()
            
            # Get available encoders
            encoders = self._get_available_encoders(serial)
            
            return {
                "serial": serial,
                "model": props.get('ro.product.model', 'Unknown'),
                "android_version": props.get('ro.build.version.release', 'Unknown'),
                "encoders": encoders,
                "encoder": encoders[0] if encoders else "unknown"
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get device info for {serial}: {e}")
            return None
    
    def _get_available_encoders(self, serial: str) -> List[str]:
        """Get available video encoders for a device"""
        import subprocess
        
        try:
            # Use MediaCodecList to get available encoders
            result = subprocess.run(['adb', '-s', serial, 'shell', 
                                   'am', 'start', '-a', 'android.intent.action.MAIN', 
                                   '-n', 'com.android.settings/.Settings'], 
                                  capture_output=True, text=True, timeout=5)
            
            # For now, return a default set of common encoders
            # In a real implementation, this would query MediaCodecList
            return ["c2.qti.hevc.encoder", "c2.qti.avc.encoder", "c2.android.hevc.encoder"]
            
        except Exception as e:
            self.logger.warning(f"Failed to get encoders for {serial}: {e}")
            return ["unknown"]
    
    def _filter_encoders(self, encoders: List[Dict]) -> List[Dict]:
        """Filter encoders based on criteria: hw accelerated, unique canonical name, video"""
        filtered = []
        canonical_names_seen = set()  # Track canonical names to avoid duplicates
        
        for encoder in encoders:
            # Check if it's an encoder (not decoder)
            if not encoder.get("is_encoder", False):
                continue
                
            # Check if it's hardware accelerated
            if not encoder.get("is_hardware_accelerated", False):
                continue
                
            # Check if it has a canonical name
            canonical_name = encoder.get("canonical_name", "")
            if not canonical_name or not canonical_name.strip():
                # Fallback to name if canonical_name is not available
                canonical_name = encoder.get("name", "")
                if not canonical_name or not canonical_name.strip():
                    continue
                    
            # Skip if we've already seen this canonical name
            if canonical_name in canonical_names_seen:
                continue
            canonical_names_seen.add(canonical_name)
                
            # Check if it's a video encoder
            mime_type = encoder.get("mime_type", "")
            if not mime_type.startswith("video/"):
                continue
                
            filtered.append(encoder)
            
        return filtered


class TestRunner:
    """Main test runner class"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_discovery = TestDiscovery()
        self.device_manager = DeviceManager(config)
        self.results = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debug information"""
        logger = logging.getLogger("ava_test_runner")
        logger.setLevel(logging.DEBUG if self.config.debug > 0 else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def discover_tests(self) -> Dict[str, List[callable]]:
        """Discover available tests"""
        return self.test_discovery.discover_tests()
    
    def list_tests(self):
        """List all available tests"""
        test_functions = self.discover_tests()
        print("Available tests:")
        for file_path, functions in test_functions.items():
            print(f"\n{file_path}:")
            for func in functions:
                print(f"  - {func.__name__}")
                
    def run_test(self, test_func: callable, device: Dict[str, Any], 
                 input_file: str, workdir: str, mediastore: str = "_mediastore") -> TestResult:
        """Run a single test"""
        start_time = time.time()
        test_name = test_func.__name__
        device_serial = device["serial"]
        
        try:
            self.logger.info(f"Running test {test_name} on device {device_serial}")
            
            # Create test-specific workdir within the main workdir
            if workdir is None:
                workdir = os.path.join(tempfile.gettempdir(), "ava_tests")
            
            # Create subdirectory for this test run
            test_workdir = os.path.join(workdir, f"{device_serial}_{test_name}")
            os.makedirs(test_workdir, exist_ok=True)
            
            # Prepare test data
            test_data = {}
            quality_metrics = {}  # Initialize quality_metrics here
            encapp, ava_common, ava_version = _import_optional_modules()
            try:
                test_data = ava_common.initialize_testdata()
            except:
                test_data = {}
            
            # Add configuration options from config
            test_data["use_yuv_encoding"] = self.config.use_yuv_encoding
            test_data["enable_device_decode"] = self.config.enable_device_decode
            test_data["mediastore"] = mediastore
            
            # Run the test function
            if self.config.dry_run:
                self.logger.info(f"Dry run: Would execute {test_name}")
                output_files = []
                test_success = True
                error_message = None
                # quality_metrics already initialized above
            else:
                # Call the test function with appropriate parameters
                self.logger.debug(f"Calling test function with input_file: {input_file}")
                result = test_func(device, input_file, test_workdir, test_data)
                
                # Check if the test function returned a result dictionary
                if isinstance(result, dict):
                    test_success = result.get("success", True)
                    output_files = result.get("output_files", [])
                    error_message = result.get("error", None)
                    
                    # Extract quality metrics from ladder results if available
                    quality_metrics = {}
                    if "ladder_results" in result:
                        # Aggregate quality metrics from all ladder results
                        all_vmaf = []
                        all_psnr = []
                        all_ssim = []
                        
                        for ladder_result in result["ladder_results"]:
                            if ladder_result.get("success") and "quality_results" in ladder_result:
                                qr = ladder_result["quality_results"]
                                if qr.get("vmaf"):
                                    all_vmaf.append(qr["vmaf"])
                                if qr.get("psnr"):
                                    all_psnr.append(qr["psnr"])
                                if qr.get("ssim"):
                                    all_ssim.append(qr["ssim"])
                        
                        # Calculate averages
                        if all_vmaf:
                            quality_metrics["vmaf"] = sum(all_vmaf) / len(all_vmaf)
                        if all_psnr:
                            quality_metrics["psnr"] = sum(all_psnr) / len(all_psnr)
                        if all_ssim:
                            quality_metrics["ssim"] = sum(all_ssim) / len(all_ssim)
                    
                    # Update test_data with any additional data from the test
                    if "test_data" in result:
                        test_data.update(result["test_data"])
                    
                    # Add quality_csv to test_data if available
                    if "quality_csv" in result:
                        test_data["quality_csv"] = result["quality_csv"]
                    
                    # Add stats_csv to test_data if available
                    if "stats_csv" in result:
                        test_data["stats_csv"] = result["stats_csv"]
                else:
                    test_success = True
                    output_files = []
                    error_message = None
                    # quality_metrics already initialized above
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                device_serial=device_serial,
                success=test_success,
                duration=duration,
                output_files=output_files,
                test_data=test_data,
                error_message=error_message,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            tb = traceback.format_exc()
            
            self.logger.error(f"Test {test_name} failed: {error_msg}")
            
            return TestResult(
                test_name=test_name,
                device_serial=device_serial,
                success=False,
                duration=duration,
                output_files=[],
                test_data={},
                error_message=error_msg,
                quality_metrics={},
                traceback=tb
            )


class IntegratedTestRunner:
    """Integrated test runner with quality assessment and reporting"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_runner = TestRunner(config)
        self.quality_assessor = QualityAssessment(debug=config.debug > 0)
        self.report_generator = ReportGenerator(debug=config.debug > 0)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debug information"""
        logger = logging.getLogger("ava_integrated")
        logger.setLevel(logging.DEBUG if self.config.debug > 0 else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def run_tests(self):
        """Run the complete test suite with source generation, quality assessment, and reporting"""
        self.logger.info("Starting AVA test execution")
        
        # Step 1: Generate or prepare video sources
        self.logger.info("Step 1: Preparing video sources")
        sources = self._prepare_sources()
        self.config.input_files = sources
        
        # Step 2: Run tests
        self.logger.info("Step 2: Running tests")
        test_results = self._run_tests()
        
        # Step 3: Generate reports
        self.logger.info("Step 4: Generating reports")
        reports = self._generate_reports(test_results, [])
        
        # Print summary
        self._print_summary(test_results, reports)
        
        return test_results, reports
    
    
    def run_quality_only(self):
        """Run quality assessment on existing test data without running new tests"""
        self.logger.info("Running quality assessment on existing test data")
        
        # Step 1: Load test results from workdir or quality_data
        if self.config.quality_data:
            test_results = self._load_test_results_from_quality_data()
        else:
            test_results = self._load_existing_test_results()
        
        if not test_results:
            self.logger.error("No test results found")
            return
        
        # Step 2: Run quality assessment on existing data
        self.logger.info("Running quality assessment on existing test results")
        test_results = self._run_quality_assessment_on_existing(test_results)
        
        # Step 3: Generate reports with quality data
        self.logger.info("Generating reports with quality data")
        reports = self._generate_reports(test_results, [])
        
        # Step 4: Print summary
        self._print_summary(test_results, reports)
        
        return test_results, reports
    
    def _run_quality_assessment_on_existing(self, test_results: List[TestResult]) -> List[TestResult]:
        """Run quality assessment on existing test results and integrate into TestResult objects"""
        
        for result in test_results:
            if result.success and result.output_files:
                # Check if quality data already exists and has content
                if (hasattr(result, 'test_data') and result.test_data and 
                    'quality_csv' in result.test_data and 
                    os.path.exists(result.test_data['quality_csv'])):
                    # Check if the CSV file has actual data (more than just headers)
                    try:
                        import pandas as pd
                        df = pd.read_csv(result.test_data['quality_csv'])
                        if not df.empty and len(df) > 0:
                            self.logger.info(f"Quality data already exists for {result.test_name} on {result.device_serial}, skipping analysis")
                            continue
                        else:
                            self.logger.info(f"Quality CSV exists but is empty for {result.test_name} on {result.device_serial}, re-running analysis")
                    except Exception as e:
                        self.logger.info(f"Quality CSV exists but is corrupted for {result.test_name} on {result.device_serial}, re-running analysis: {e}")
                
                self.logger.info(f"Running quality assessment for {result.test_name} on {result.device_serial}")
                
                # Find JSON files in the output
                json_files = [f for f in result.output_files if f.endswith('.json')]
                if json_files:
                    self.logger.info(f"Found {len(json_files)} JSON files for quality assessment")
                    
                    try:
                        # Use ava_common.run_encapp_quality to process all JSON files
                        from ava import ava_common
                        output_dir = os.path.dirname(json_files[0])
                        csv_output = os.path.join(output_dir, "quality_analysis.csv")
                        
                        # Run encapp_quality on all JSON files
                        success = ava_common.run_encapp_quality(
                            json_files, 
                            output_dir, 
                            self.config.mediastore, 
                            max_parallel=1
                        )
                        
                        if success and os.path.exists(csv_output):
                            # Parse the quality CSV and integrate into TestResult
                            quality_data_list = self._parse_quality_csv(csv_output)
                            if quality_data_list:
                                # Integrate quality data directly into TestResult
                                result.quality_metrics = quality_data_list
                                if result.test_data is None:
                                    result.test_data = {}
                                result.test_data["quality_csv"] = csv_output
                                self.logger.info(f"Successfully integrated {len(quality_data_list)} quality data points into {result.test_name}")
                            else:
                                self.logger.warning(f"Failed to parse quality data from {csv_output}")
                        else:
                            self.logger.warning(f"Failed to generate quality data for {result.test_name}")
                        
                        # Generate performance CSV files from JSON files
                        self.logger.info(f"Generating performance CSV files for {result.test_name}")
                        try:
                            performance_csv_files = ava_common.run_encapp_stats_to_csv(json_files, output_dir)
                            if performance_csv_files:
                                if result.test_data is None:
                                    result.test_data = {}
                                result.test_data["stats_csv"] = performance_csv_files
                                self.logger.info(f"Successfully generated {len(performance_csv_files)} performance CSV files for {result.test_name}")
                            else:
                                self.logger.warning(f"No performance CSV files generated for {result.test_name}")
                        except Exception as e:
                            self.logger.error(f"Error generating performance CSV files for {result.test_name}: {e}")
                            
                    except Exception as e:
                        self.logger.error(f"Error running quality assessment for {result.test_name}: {e}")
                else:
                    self.logger.warning(f"No JSON files found for {result.test_name}")
        
        return test_results
    
    def _run_encapp_quality_for_file(self, json_file: str, test_name: str) -> Optional[str]:
        """Run encapp_quality on a specific JSON file"""
        try:
            import subprocess
            import os
            
            # Determine output directory
            output_dir = os.path.dirname(json_file)
            csv_output = os.path.join(output_dir, "quality_analysis.csv")
            
            # Run encapp_quality
            cmd = [
                "python3", "-m", "encapp_quality",
                json_file,
                "--media", self.config.mediastore,
                "--output", csv_output,
                "--keep-quality-files",
                "--ignore-timing",
                "--header"
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/johanblome/code/ava")
            
            if result.returncode == 0:
                if os.path.exists(csv_output):
                    self.logger.info(f"Quality assessment completed: {csv_output}")
                    return csv_output
                else:
                    self.logger.error(f"Quality assessment completed but output file not found: {csv_output}")
            else:
                self.logger.error(f"Quality assessment failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error running encapp_quality: {e}")
        
        return None
    
    def _parse_quality_csv(self, csv_file: str) -> Optional[List[Dict[str, Any]]]:
        """Parse quality CSV file and return list of quality metrics"""
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_file)
            if df.empty:
                return None
            
            # Extract quality metrics from each row in the CSV
            quality_results = []
            for _, row in df.iterrows():
                metrics = {
                    # Basic file info
                    "media": row.get('media', None),
                    "description": row.get('description', None),
                    "id": row.get('id', None),
                    "testfile": row.get('testfile', None),
                    "reference_file": row.get('reference_file', None),
                    
                    # Device/platform info
                    "model": row.get('model', None),
                    "platform": row.get('platform', None),
                    "serial": row.get('serial', None),
                    
                    # Codec and encoding settings
                    "codec": row.get('codec', None),
                    "bitrate_mode": row.get('bitrate_mode', None),
                    "quality": row.get('quality', None),
                    "gop_sec": row.get('gop_sec', None),
                    
                    # Video properties
                    "framerate_fps": row.get('framerate_fps', None),
                    "width": row.get('width', None),
                    "height": row.get('height', None),
                    "resolution": f"{row.get('width', 0)}x{row.get('height', 0)}" if row.get('width') and row.get('height') else None,
                    
                    # Bitrate metrics
                    "bitrate_bps": row.get('bitrate_bps', None),
                    "bitrate": row.get('bitrate_bps', None),  # Alias for compatibility
                    "meanbitrate_bps": row.get('meanbitrate_bps', None),
                    "mean_bpp": row.get('mean_bpp', None),
                    "calculated_bitrate_bps": row.get('calculated_bitrate_bps', None),
                    
                    # Frame info
                    "framecount": row.get('framecount', None),
                    "duration": row.get('framecount', None),  # Using framecount as duration proxy
                    "iframes": row.get('iframes', None),
                    "pframes": row.get('pframes', None),
                    "bframes": row.get('bframes', None),
                    "iframe_size_bytes": row.get('iframe_size_bytes', None),
                    "pframe_size_bytes": row.get('pframe_size_bytes', None),
                    "bframe_size_bytes": row.get('bframe_size_bytes', None),
                    
                    # File size
                    "size_bytes": row.get('size_bytes', None),
                    "file_size": row.get('size_bytes', None),  # Alias for compatibility
                    
                    # VMAF metrics (using correct column names)
                    "vmaf_mean": row.get('vmaf_mean', None),
                    "vmaf": row.get('vmaf_mean', None),  # Alias for compatibility
                    "vmaf_harmonic_mean": row.get('vmaf_harmonic_mean', None),
                    "vmaf_min": row.get('vmaf_min', None),
                    "vmaf_max": row.get('vmaf_max', None),
                    "vmaf_p5": row.get('vmaf_p5', None),
                    "vmaf_p10": row.get('vmaf_p10', None),
                    "vmaf_p25": row.get('vmaf_p25', None),
                    "vmaf_p50": row.get('vmaf_p50', None),
                    "vmaf_p75": row.get('vmaf_p75', None),
                    "vmaf_p90": row.get('vmaf_p90', None),
                    "vmaf_p95": row.get('vmaf_p95', None),
                    "vmaf_model": row.get('vmaf_model', None),
                    "vmaf_framecount": row.get('vmaf_framecount', None),
                    "vmaf_zero_vmaf": row.get('vmaf_zero_vmaf', None),
                    
                    # Quality metrics (using correct column names)
                    "ssim": row.get('ssim', None),
                    "psnr": row.get('psnr', None),
                    "psnr_y": row.get('psnr_y', None),
                    "psnr_u": row.get('psnr_u', None),
                    "psnr_v": row.get('psnr_v', None),
                    "cvvdp": row.get('cvvdp', None),
                    
                    # QP metrics (using correct column names with _min/_max/_avg)
                    "qpy_min": row.get('qpy_min', None),
                    "qpy_max": row.get('qpy_max', None),
                    "qpy_avg": row.get('qpy_avg', None),
                    "qpu_min": row.get('qpu_min', None),
                    "qpu_max": row.get('qpu_max', None),
                    "qpu_avg": row.get('qpu_avg', None),
                    "qpv_min": row.get('qpv_min', None),
                    "qpv_max": row.get('qpv_max', None),
                    "qpv_avg": row.get('qpv_avg', None),
                    
                    # Complexity metrics (using correct column names with _min/_max/_avg)
                    "si_min": row.get('si_min', None),
                    "si_max": row.get('si_max', None),
                    "si_avg": row.get('si_avg', None),
                    "ti_min": row.get('ti_min', None),
                    "ti_max": row.get('ti_max', None),
                    "ti_avg": row.get('ti_avg', None),
                    
                    # Source analysis
                    "source_complexity": row.get('source_complexity', None),
                    "source_motions": row.get('source_motions', None),
                    "warning": row.get('warning', None)
                }
                # Only add non-null values
                quality_metrics = {k: v for k, v in metrics.items() if v is not None}
                if quality_metrics:  # Only add if we have some data
                    quality_results.append(quality_metrics)
            
            return quality_results if quality_results else None
            
        except Exception as e:
            self.logger.error(f"Failed to parse quality CSV {csv_file}: {e}")
            return None
    
    def _load_existing_test_results(self) -> List[TestResult]:
        """Load existing test results from the workdir"""
        test_results = []
        
        if not os.path.exists(self.config.workdir):
            self.logger.error(f"Workdir does not exist: {self.config.workdir}")
            self.logger.error("Make sure you're using the same workdir path that was used during test execution.")
            self.logger.error("You can find existing test data with: find /tmp -name '*ava*' -type d")
            return test_results
        
        # Look for test result directories
        for item in os.listdir(self.config.workdir):
            item_path = os.path.join(self.config.workdir, item)
            if os.path.isdir(item_path) and '_' in item:
                # Try to extract device serial and test name from directory name
                parts = item.split('_')
                if len(parts) >= 2:
                    device_serial = parts[0]
                    test_name = '_'.join(parts[1:])
                    
                    # Look for output files, quality CSV, and stats CSV files
                    output_files = []
                    quality_csv = None
                    stats_csv_files = []
                    
                    for file in os.listdir(item_path):
                        file_path = os.path.join(item_path, file)
                        if file.endswith('.json'):
                            output_files.append(file_path)
                        elif file == 'quality_analysis.csv':
                            quality_csv = file_path
                        elif file.endswith('_encoding_data.csv'):
                            stats_csv_files.append(file_path)
                    
                    if output_files:
                        # Create TestResult from existing data
                        test_data = {}
                        if quality_csv:
                            test_data["quality_csv"] = quality_csv
                        if stats_csv_files:
                            test_data["stats_csv"] = stats_csv_files
                        
                        test_result = TestResult(
                            test_name=test_name,
                            device_serial=device_serial,
                            success=True,
                            duration=0,  # Unknown duration for existing results
                            output_files=output_files,
                            test_data=test_data,
                            error_message=None,
                            quality_metrics={}
                        )
                        test_results.append(test_result)
                        self.logger.info(f"Loaded existing test result: {test_name} on {device_serial}")
        
        return test_results
    
    def _load_test_results_from_quality_data(self) -> List[TestResult]:
        """Load test results from directly specified quality CSV files"""
        test_results = []
        
        for i, quality_csv in enumerate(self.config.quality_data):
            if not os.path.exists(quality_csv):
                self.logger.warning(f"Quality CSV file not found: {quality_csv}")
                continue
            
            self.logger.info(f"Loading quality data from: {quality_csv}")
            
            # Find corresponding encoding data CSV in the same directory
            quality_dir = os.path.dirname(quality_csv)
            encoding_csv_files = []
            
            # Look for _encoding_data.csv files in the same directory
            # Handle case where quality_csv is in current directory (quality_dir is empty)
            if quality_dir:
                try:
                    for file in os.listdir(quality_dir):
                        if file.endswith('_encoding_data.csv'):
                            encoding_csv_files.append(os.path.join(quality_dir, file))
                except (OSError, FileNotFoundError):
                    # Directory doesn't exist or can't be read, skip encoding data search
                    pass
            else:
                # CSV file is in current directory, look there
                try:
                    for file in os.listdir('.'):
                        if file.endswith('_encoding_data.csv'):
                            encoding_csv_files.append(file)
                except (OSError, FileNotFoundError):
                    # Current directory can't be read, skip encoding data search
                    pass
            
            # Create TestResult from quality CSV
            test_data = {
                "quality_csv": quality_csv
            }
            
            if encoding_csv_files:
                test_data["stats_csv"] = encoding_csv_files
                self.logger.info(f"Found {len(encoding_csv_files)} encoding data files")
            
            # Extract test info from filename or CSV content
            test_name = "quality_analysis"
            device_serial = "unknown"
            
            # Use custom label if provided, otherwise extract from CSV content
            if (self.config.quality_data_labels and 
                i < len(self.config.quality_data_labels) and 
                self.config.quality_data_labels[i]):
                # Use custom label for test name
                test_name = f"quality_analysis_{self.config.quality_data_labels[i]}"
                self.logger.info(f"Using custom label: {self.config.quality_data_labels[i]}")
            else:
                # Try to extract info from CSV content (original behavior)
                try:
                    import pandas as pd
                    df = pd.read_csv(quality_csv)
                    if not df.empty:
                        # Get unique codecs and devices from the CSV
                        codecs = df['codec'].unique() if 'codec' in df.columns else ['unknown']
                        devices = df['serial'].unique() if 'serial' in df.columns else ['unknown']
                        
                        # Use the first codec and device for naming
                        if len(codecs) > 0 and codecs[0] != 'unknown':
                            test_name = f"quality_analysis_{codecs[0]}"
                        if len(devices) > 0 and devices[0] != 'unknown':
                            device_serial = devices[0]
                            
                except Exception as e:
                    self.logger.warning(f"Could not extract metadata from {quality_csv}: {e}")
            
            test_result = TestResult(
                test_name=test_name,
                device_serial=device_serial,
                success=True,
                duration=0,  # Unknown duration for existing results
                output_files=[],  # No JSON files for direct CSV input
                test_data=test_data,
                error_message=None,
                quality_metrics={}
            )
            test_results.append(test_result)
            self.logger.info(f"Loaded quality data: {test_name} on {device_serial}")
        
        return test_results
    
    def _prepare_sources(self) -> List[str]:
        """Prepare video sources for testing"""
        # If input files are explicitly provided, use only those
        if self.config.input_files:
            self.logger.info(f"Using provided input files: {self.config.input_files}")
            return self.config.input_files
        
        # For now, require input files to be provided
        # In the future, this could scan a directory for video files
        self.logger.error("No input files provided. Please specify video files using --input-file")
        return []
    
    def _run_tests(self) -> List[TestResult]:
        """Run the actual tests"""
        # Use the test runner to discover and run tests
        test_functions = self.test_runner.discover_tests()
        if not test_functions:
            self.logger.warning("No tests found")
            return []
        
        # Discover devices (use cached results if available)
        if not hasattr(self, '_cached_devices'):
            self._cached_devices = self.test_runner.device_manager.discover_devices()
        devices = self._cached_devices
        self.logger.debug(f"Discovered {len(devices)} devices: {[d.get('serial', 'unknown') for d in devices]}")
        if not devices:
            self.logger.error("No suitable devices found")
            return []
        
        # Filter tests if specific test requested
        if self.config.test_name:
            test_func = self.test_runner.test_discovery.get_test_by_name(self.config.test_name)
            if not test_func:
                self.logger.error(f"Test '{self.config.test_name}' not found")
                return []
            test_functions = {f"custom": [test_func]}
        
        # Run tests
        all_results = []
        total_tests = sum(len(functions) for functions in test_functions.values()) * len(devices) * len(self.config.input_files)
        
        self.logger.info(f"Running {total_tests} test combinations...")
        
        for file_path, functions in test_functions.items():
            for test_func in functions:
                for device in devices:
                    for input_file in self.config.input_files:
                        self.logger.debug(f"Running test {test_func.__name__} with input_file: {input_file}")
                        try:
                            result = self.test_runner.run_test(test_func, device, input_file, self.config.workdir, self.config.mediastore)
                            all_results.append(result)
                        except Exception as e:
                            self.logger.error(f"Test {test_func.__name__} failed: {e}")
                            error_result = TestResult(
                                test_name=test_func.__name__,
                                device_serial=device["serial"],
                                success=False,
                                duration=0,
                                output_files=[],
                                test_data={},
                                error_message=str(e),
                                traceback=traceback.format_exc()
                            )
                            all_results.append(error_result)
        
        return all_results
    
    
    def _generate_reports(self, test_results: List[TestResult], 
                         quality_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate comprehensive reports"""
        # Convert test results to report format
        report_test_results = []
        for result in test_results:
            report_test_results.append(ReportTestResult(
                test_name=result.test_name,
                device_serial=result.device_serial,
                encoder=getattr(result, 'encoder', 'unknown'),
                success=result.success,
                duration=result.duration,
                quality_metrics=result.quality_metrics,
                test_data=result.test_data,
                error_message=result.error_message
            ))
        
        # Generate reports
        reports = self.report_generator.generate_comprehensive_report(
            report_test_results, quality_results
        )
        
        return reports
    
    def _print_summary(self, test_results: List[TestResult], 
                      reports: Dict[str, str]):
        """Print execution summary"""
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r.success)
        failed_tests = total_tests - successful_tests
        
        print(f"\n{'='*60}")
        print(f"AVA Test Execution Summary")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        
        # Extract quality data from integrated TestResult objects
        all_quality_data = []
        for result in test_results:
            if result.quality_metrics:
                all_quality_data.extend(result.quality_metrics)
        
        if all_quality_data:
            print(f"Quality Assessments: {len(all_quality_data)}")
            psnr_values = [r.get("psnr") for r in all_quality_data if r.get("psnr") is not None]
            if psnr_values:
                print(f"Average PSNR: {sum(psnr_values)/len(psnr_values):.2f} dB")
        
        print(f"\nReports Generated:")
        for report_type, report_path in reports.items():
            if report_path:
                print(f"  {report_type}: {report_path}")
        
        if failed_tests > 0:
            print(f"\nFailed Tests:")
            for result in test_results:
                if not result.success:
                    print(f"  - {result.test_name} on {result.device_serial}: {result.error_message}")
        
        print(f"{'='*60}")


def get_options(argv):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AVA Test Runner - Video Codec Testing Framework"
    )
    
    parser.add_argument(
        "-v", "--version", action="version", version="2.0.0"
    )
    parser.add_argument(
        "-d", "--debug", action="count", default=0,
        help="Increase verbosity (use multiple times for more)"
    )
    parser.add_argument(
        "--quiet", action="store_const", dest="debug", const=-1,
        help="Zero verbosity"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Dry run - don't execute tests"
    )
    parser.add_argument(
        "-s", "--serial", action="append", dest="android_serials",
        help="Device serial number (can be used multiple times)"
    )
    parser.add_argument(
        "--encoder", type=str, help="Encoder name (partial or full match)"
    )
    parser.add_argument(
        "--test", type=str, help="Run specific test by name"
    )
    parser.add_argument(
        "--list-tests", action="store_true", help="List all available tests"
    )
    parser.add_argument(
        "--input-file", action="append", dest="input_files",
        help="Input video file (can be used multiple times)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for results"
    )
    parser.add_argument(
        "--workdir", type=str, help="Working directory for test execution"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--run-quality-only", action="store_true",
        help="Run quality assessment on existing test data without running new tests"
    )
    parser.add_argument(
        "--generate-reports-only", action="store_true",
        help="Generate reports from existing test data without running new tests"
    )
    parser.add_argument(
        "--use-yuv-encoding", action="store_true", default=False,
        help="Use YUV encoding instead of MP4 transcoding"
    )
    parser.add_argument(
        "--disable-device-decode", action="store_true", default=False,
        help="Disable on-device decoding (default: enabled)"
    )
    parser.add_argument(
        "--mediastore", type=str, default="_mediastore",
        help="Directory for storing reference videos and media files (default: _mediastore)"
    )
    parser.add_argument(
        "--quality-data", type=str,
        help="Comma-separated list of quality CSV files to analyze (alternative to --workdir)"
    )
    parser.add_argument(
        "--quality-data-labels", type=str,
        help="Comma-separated list of labels for quality CSV files (must match order of --quality-data files)"
    )
    
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    """Main entry point"""
    options = get_options(argv)
    
    # Create timestamped workdir if none provided and not using quality_data
    if options.workdir is None and not options.quality_data:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        options.workdir = os.path.join(tempfile.gettempdir(), f"ava_tests_{timestamp}")

    # Parse quality_data if provided
    quality_data = []
    if options.quality_data:
        quality_data = [f.strip() for f in options.quality_data.split(',') if f.strip()]
    
    # Parse quality_data_labels if provided
    quality_data_labels = []
    if options.quality_data_labels:
        quality_data_labels = [f.strip() for f in options.quality_data_labels.split(',') if f.strip()]
        
        # Validate that labels match quality_data count
        if quality_data and len(quality_data_labels) != len(quality_data):
            print(f"Error: Number of quality-data-labels ({len(quality_data_labels)}) must match number of quality-data files ({len(quality_data)})")
            print(f"Quality data files: {quality_data}")
            print(f"Quality data labels: {quality_data_labels}")
            sys.exit(1)
    
    # Create configuration
    config = TestConfig(
        debug=options.debug,
        dry_run=options.dry_run,
        android_serials=options.android_serials or [],
        encoder=options.encoder,
        test_name=options.test,
        test_list=options.list_tests,
        input_files=options.input_files,
        output_dir=options.output_dir,
        workdir=options.workdir,
        max_workers=options.max_workers,
        mediastore=options.mediastore,
        use_yuv_encoding=options.use_yuv_encoding,
        enable_device_decode=not options.disable_device_decode,
        quality_data=quality_data,
        quality_data_labels=quality_data_labels
    )
    
    # Handle special commands
    if options.list_tests:
        runner = TestRunner(config)
        runner.list_tests()
        return
    
    
    if options.run_quality_only:
        # Run quality assessment on existing test data
        integrated_runner = IntegratedTestRunner(config)
        integrated_runner.run_quality_only()
        return
    
    if options.generate_reports_only:
        # Generate reports from existing test data
        integrated_runner = IntegratedTestRunner(config)
        
        # Load existing test results
        if config.quality_data:
            test_results = integrated_runner._load_test_results_from_quality_data()
        else:
            test_results = integrated_runner._load_existing_test_results()
        
        if not test_results:
            print("❌ No test results found. Please run tests first or specify a valid workdir.")
            return
        
        # Generate reports
        from .ava_report import ReportGenerator
        report_generator = ReportGenerator(debug=config.debug > 0)
        reports = report_generator.generate_comprehensive_report(test_results)
        
        print(f"📊 Reports generated:")
        for report_type, report_path in reports.items():
            print(f"   {report_type}: {report_path}")
        
        return
    
    
    # Create integrated test runner
    runner = IntegratedTestRunner(config)
    
    try:
        # Run the complete test suite
        test_results, reports = runner.run_tests()
        
        # Exit with appropriate code
        failed_tests = sum(1 for r in test_results if not r.success)
        if failed_tests > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        if config.debug > 0:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
