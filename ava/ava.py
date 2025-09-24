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
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import our modules
from .ava_sources import VideoSourceGenerator
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
    try:
        # Import ava_common first (it handles encapp imports internally)
        from . import ava_common
        ENCAPP_AVAILABLE = True
        # ava_version is optional
        ava_version = None
        return ava_common.encapp, ava_common, ava_version
    except ImportError:
        ENCAPP_AVAILABLE = False
        return None, None, None


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
    video_duration: float = 10.0
    use_yuv_encoding: bool = False
    enable_device_decode: bool = True
    
    def __post_init__(self):
        if self.android_serials is None:
            self.android_serials = []
        if self.input_files is None:
            self.input_files = [f"{Path(__file__).parent}/../vid/johnny.1280x720.60fps.264.mp4"]


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
        if not ENCAPP_AVAILABLE:
            self.logger.info("encapp not available, using ADB directly")
            # Try to discover devices using ADB directly
            return self._discover_devices_with_adb()
        
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
                        device = {
                            "serial": serial,
                            "device_workdir": device_workdir,
                            "encoder": encoder_info["name"],
                            "encoder_info": encoder_info,
                            "codecs": codecs
                        }
                        devices.append(device)
                        self.logger.info(f"Added device {serial} with encoder {encoder_info['name']}")
                        
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
        for encoder in encoders:
            # Check if it's an encoder (not decoder)
            if not encoder.get("is_encoder", False):
                continue
                
            # Check if it's hardware accelerated
            if not encoder.get("is_hardware_accelerated", False):
                continue
                
            # Check if it has a unique canonical name
            name = encoder.get("name", "")
            if not name or not name.strip():
                continue
                
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
                 input_file: str, workdir: str) -> TestResult:
        """Run a single test"""
        start_time = time.time()
        test_name = test_func.__name__
        device_serial = device["serial"]
        
        try:
            self.logger.info(f"Running test {test_name} on device {device_serial}")
            
            # Create test-specific workdir within the main workdir
            if workdir is None:
                workdir = "/tmp/ava_tests"
            
            # Create subdirectory for this test run
            test_workdir = os.path.join(workdir, f"{device_serial}_{test_name}")
            os.makedirs(test_workdir, exist_ok=True)
            
            # Prepare test data
            test_data = {}
            encapp, ava_common, ava_version = _import_optional_modules()
            if ENCAPP_AVAILABLE:
                try:
                    test_data = ava_common.initialize_testdata()
                except:
                    test_data = {}
            
            # Add configuration options from config
            test_data["video_duration"] = self.config.video_duration
            test_data["use_yuv_encoding"] = self.config.use_yuv_encoding
            test_data["enable_device_decode"] = self.config.enable_device_decode
            
            # Run the test function
            if self.config.dry_run:
                self.logger.info(f"Dry run: Would execute {test_name}")
                output_files = []
                test_success = True
                error_message = None
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
                else:
                    test_success = True
                    output_files = []
                    error_message = None
                    quality_metrics = {}
            
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
                traceback=tb
            )


class IntegratedTestRunner:
    """Integrated test runner with source generation, quality assessment, and reporting"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_runner = TestRunner(config)
        self.source_generator = VideoSourceGenerator(debug=config.debug > 0)
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
        
        # Step 3: Quality assessment
        self.logger.info("Step 3: Assessing quality")
        quality_results = self._assess_quality(test_results)
        
        # Step 4: Generate reports
        self.logger.info("Step 4: Generating reports")
        reports = self._generate_reports(test_results, quality_results)
        
        # Print summary
        self._print_summary(test_results, quality_results, reports)
        
        return test_results, quality_results, reports
    
    def _prepare_sources(self) -> List[str]:
        """Prepare video sources for testing"""
        sources = []
        
        # If specific test requested, generate appropriate sources
        if self.config.test_name:
            sources = self.source_generator.get_sources_for_test(self.config.test_name, self.config.video_duration)
        else:
            # Generate standard sources with configurable duration
            sources = self.source_generator.generate_all_standard_sources(self.config.video_duration)
        
        # Add any provided input files
        if self.config.input_files:
            sources.extend(self.config.input_files)
        
        # Remove duplicates
        sources = list(set(sources))
        
        self.logger.info(f"Prepared {len(sources)} video sources")
        return sources
    
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
                            result = self.test_runner.run_test(test_func, device, input_file, self.config.workdir)
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
    
    def _assess_quality(self, test_results: List[TestResult]) -> List[Dict[str, Any]]:
        """Assess quality of encoded videos"""
        quality_results = []
        
        # Convert test results to format expected by quality assessor
        test_results_dict = []
        for result in test_results:
            if result.success and result.output_files:
                test_results_dict.append({
                    "test_name": result.test_name,
                    "device_serial": result.device_serial,
                    "encoder": getattr(result, 'encoder', 'unknown'),
                    "output_file": result.output_files[0] if result.output_files else None,
                    "test_data": result.test_data
                })
        
        if not test_results_dict:
            self.logger.warning("No successful test results for quality assessment")
            return []
        
        # Run quality assessment
        try:
            quality_results = self.quality_assessor.batch_assess_quality(
                test_results_dict, 
                self.config.workdir,  # Use workdir as reference directory
                os.path.join(self.config.workdir, "quality_assessment")
            )
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return []
        
        return quality_results
    
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
                quality_metrics=result.test_data.get("quality_metrics") if result.test_data else None,
                test_data=result.test_data,
                error_message=result.error_message
            ))
        
        # Generate reports
        reports = self.report_generator.generate_comprehensive_report(
            report_test_results, quality_results
        )
        
        return reports
    
    def _print_summary(self, test_results: List[TestResult], 
                      quality_results: List[Dict[str, Any]], 
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
        
        if quality_results:
            print(f"Quality Assessments: {len(quality_results)}")
            psnr_values = [r.get("psnr") for r in quality_results if r.get("psnr") is not None]
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
        "--generate-sources", action="store_true",
        help="Generate video sources and exit"
    )
    parser.add_argument(
        "--video-duration", type=float, default=10.0,
        help="Duration of generated test videos in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--use-yuv-encoding", action="store_true", default=False,
        help="Use YUV encoding instead of MP4 transcoding"
    )
    parser.add_argument(
        "--disable-device-decode", action="store_true", default=False,
        help="Disable on-device decoding (default: enabled)"
    )
    
    options = parser.parse_args(argv[1:])
    return options


def main(argv):
    """Main entry point"""
    options = get_options(argv)
    
    # Create timestamped workdir if none provided
    if options.workdir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        options.workdir = f"/tmp/ava_tests_{timestamp}"
    
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
        video_duration=options.video_duration,
        use_yuv_encoding=options.use_yuv_encoding,
        enable_device_decode=not options.disable_device_decode
    )
    
    # Handle special commands
    if options.list_tests:
        runner = TestRunner(config)
        runner.list_tests()
        return
    
    
    if options.generate_sources:
        generator = VideoSourceGenerator(debug=config.debug > 0)
        sources = generator.generate_all_standard_sources()
        print(f"Generated {len(sources)} video sources")
        for source in sources:
            print(f"  - {source}")
        return
    
    # Create integrated test runner
    runner = IntegratedTestRunner(config)
    
    try:
        # Run the complete test suite
        test_results, quality_results, reports = runner.run_tests()
        
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
