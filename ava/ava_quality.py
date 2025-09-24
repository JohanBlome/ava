#!/usr/bin/env python3

"""
AVA Quality Assessment Module

This module integrates encapp_quality.py for quality assessment of encoded videos.
It provides a unified interface for running quality metrics and analysis.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class QualityMetrics:
    """Container for quality assessment metrics"""
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    vmaf: Optional[float] = None
    bitrate: Optional[float] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    framerate: Optional[float] = None
    codec: Optional[str] = None
    encoder: Optional[str] = None
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class QualityAssessment:
    """Quality assessment using encapp_quality.py and other tools"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debug information"""
        logger = logging.getLogger("ava_quality")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def assess_quality(self, reference_video: str, encoded_video: str, 
                      output_dir: str = None) -> QualityMetrics:
        """
        Assess quality of encoded video against reference
        
        Args:
            reference_video: Path to reference video file
            encoded_video: Path to encoded video file
            output_dir: Directory for quality assessment outputs
            
        Returns:
            QualityMetrics object with assessment results
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="ava_quality_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Assessing quality: {encoded_video} vs {reference_video}")
        
        metrics = QualityMetrics()
        
        try:
            # Get basic video information
            ref_info = self._get_video_info(reference_video)
            enc_info = self._get_video_info(encoded_video)
            
            metrics.resolution = enc_info.get("resolution")
            metrics.framerate = enc_info.get("framerate")
            metrics.duration = enc_info.get("duration")
            metrics.file_size = enc_info.get("file_size")
            metrics.bitrate = enc_info.get("bitrate")
            metrics.codec = enc_info.get("codec")
            
            # Run PSNR calculation
            metrics.psnr = self._calculate_psnr(reference_video, encoded_video, output_dir)
            
            # Run SSIM calculation
            metrics.ssim = self._calculate_ssim(reference_video, encoded_video, output_dir)
            
            # Run VMAF calculation (if available)
            metrics.vmaf = self._calculate_vmaf(reference_video, encoded_video, output_dir)
            
            # Run encapp_quality if available
            encapp_metrics = self._run_encapp_quality(reference_video, encoded_video, output_dir)
            if encapp_metrics:
                metrics.additional_metrics.update(encapp_metrics)
            
            self.logger.info(f"Quality assessment completed: PSNR={metrics.psnr:.2f}, "
                           f"SSIM={metrics.ssim:.4f}, VMAF={metrics.vmaf:.2f}")
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            raise
        
        return metrics
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get basic video information using ffprobe"""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Calculate bitrate
            format_info = data.get("format", {})
            file_size = int(format_info.get("size", 0))
            duration = float(format_info.get("duration", 0))
            bitrate = (file_size * 8) / duration if duration > 0 else 0
            
            return {
                "resolution": (video_stream.get("width", 0), video_stream.get("height", 0)),
                "framerate": eval(video_stream.get("r_frame_rate", "0/1")),
                "duration": duration,
                "file_size": file_size,
                "bitrate": bitrate,
                "codec": video_stream.get("codec_name", "unknown")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get video info for {video_path}: {e}")
            return {}
    
    def _calculate_psnr(self, reference: str, encoded: str, output_dir: str) -> Optional[float]:
        """Calculate PSNR using FFmpeg"""
        psnr_file = os.path.join(output_dir, "psnr.txt")
        
        cmd = [
            "ffmpeg", "-i", encoded, "-i", reference,
            "-lavfi", "psnr=stats_file=" + psnr_file,
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse PSNR from output
            for line in result.stderr.split('\n'):
                if 'psnr_avg' in line:
                    # Extract PSNR value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'psnr_avg' in part and i + 1 < len(parts):
                            return float(parts[i + 1].split(':')[1])
            
            # Fallback: read from stats file
            if os.path.exists(psnr_file):
                with open(psnr_file, 'r') as f:
                    for line in f:
                        if 'psnr_avg' in line:
                            return float(line.split()[1])
            
        except Exception as e:
            self.logger.warning(f"PSNR calculation failed: {e}")
        
        return None
    
    def _calculate_ssim(self, reference: str, encoded: str, output_dir: str) -> Optional[float]:
        """Calculate SSIM using FFmpeg"""
        ssim_file = os.path.join(output_dir, "ssim.txt")
        
        cmd = [
            "ffmpeg", "-i", encoded, "-i", reference,
            "-lavfi", "ssim=stats_file=" + ssim_file,
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse SSIM from output
            for line in result.stderr.split('\n'):
                if 'SSIM' in line and 'All:' in line:
                    # Extract SSIM value
                    parts = line.split('All:')[1].split()
                    if parts:
                        return float(parts[0])
            
            # Fallback: read from stats file
            if os.path.exists(ssim_file):
                with open(ssim_file, 'r') as f:
                    for line in f:
                        if 'SSIM' in line and 'All:' in line:
                            return float(line.split('All:')[1].split()[0])
            
        except Exception as e:
            self.logger.warning(f"SSIM calculation failed: {e}")
        
        return None
    
    def _calculate_vmaf(self, reference: str, encoded: str, output_dir: str) -> Optional[float]:
        """Calculate VMAF using FFmpeg (if VMAF model is available)"""
        vmaf_file = os.path.join(output_dir, "vmaf.json")
        
        cmd = [
            "ffmpeg", "-i", encoded, "-i", reference,
            "-lavfi", f"libvmaf=log_path={vmaf_file}:log_fmt=json",
            "-f", "null", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse VMAF from JSON output
            if os.path.exists(vmaf_file):
                with open(vmaf_file, 'r') as f:
                    data = json.load(f)
                    frames = data.get("frames", [])
                    if frames:
                        vmaf_scores = [frame.get("metrics", {}).get("vmaf", 0) for frame in frames]
                        return sum(vmaf_scores) / len(vmaf_scores) if vmaf_scores else None
            
        except Exception as e:
            self.logger.warning(f"VMAF calculation failed: {e}")
        
        return None
    
    def _run_encapp_quality(self, test_result_file: str, reference_dir: str, output_dir: str) -> Optional[Dict[str, Any]]:
        """Run encapp_quality.py on a test result file"""
        try:
            # Look for encapp_quality.py in the lib/encapp directory
            encapp_quality_path = None
            possible_paths = [
                "lib/encapp/scripts/encapp_quality.py",
                "../lib/encapp/scripts/encapp_quality.py",
                "encapp_quality.py"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    encapp_quality_path = path
                    break
            
            if not encapp_quality_path:
                self.logger.debug("encapp_quality.py not found, skipping")
                return None
            
            # Calculate max parallel processes (number of cores - 1)
            import multiprocessing
            max_parallel = max(1, multiprocessing.cpu_count() - 1)
            
            # Run encapp_quality on the test result file with enhanced options
            quality_output = os.path.join(output_dir, "quality.csv")
            cmd = [
                "python3", encapp_quality_path,
                test_result_file,
                "--output", quality_output,
                "--media", reference_dir,
                "--header",  # Add header to CSV data
                "--keep-quality-files",  # Keep intermediate results for time series plotting
                "--ignore-timing",  # Ignore timing information for frame-by-frame comparison
                "--max-parallel", str(max_parallel)  # Use cores - 1 for parallel processing
            ]
            
            self.logger.info(f"Running encapp_quality with {max_parallel} parallel processes: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse CSV results
            if os.path.exists(quality_output):
                return self._parse_quality_csv(quality_output)
            
        except Exception as e:
            self.logger.warning(f"encapp_quality failed: {e}")
        
        return None
    
    def _parse_quality_csv(self, csv_file: str) -> Optional[Dict[str, Any]]:
        """Parse quality CSV file from encapp_quality"""
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_file)
            if df.empty:
                return None
            
            # Get the first row (assuming single test)
            row = df.iloc[0]
            
            # Extract quality metrics
            metrics = {}
            if 'psnr_avg' in df.columns:
                metrics['psnr'] = float(row['psnr_avg'])
            if 'ssim' in df.columns:
                metrics['ssim'] = float(row['ssim'])
            if 'vmaf' in df.columns:
                metrics['vmaf'] = float(row['vmaf'])
            if 'bitrate' in df.columns:
                metrics['bitrate'] = float(row['bitrate'])
            if 'file_size' in df.columns:
                metrics['file_size'] = int(row['file_size'])
            if 'duration' in df.columns:
                metrics['duration'] = float(row['duration'])
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to parse quality CSV: {e}")
            return None
    
    def run_quality_assessment(self, device: Dict[str, Any], files: List[str], 
                              testdata: Dict[str, Any], workdir: str, 
                              test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run quality assessment for a single test result
        
        Args:
            device: Device information
            files: List of output files from encoding (includes JSON result files)
            testdata: Test data from encapp
            workdir: Working directory
            test_data: Additional test data
            
        Returns:
            Dictionary with quality metrics
        """
        if not files:
            self.logger.warning("No output files for quality assessment")
            return {}
        
        # Find the JSON result file (should be the first file)
        json_result_file = files[0]
        if not os.path.exists(json_result_file):
            self.logger.warning(f"JSON result file not found: {json_result_file}")
            return {}
        
        # Find reference file
        reference_file = testdata.get("sourcefile")
        if not reference_file or not os.path.exists(reference_file):
            self.logger.warning(f"Reference file not found: {reference_file}")
            return {}
        
        # Create quality assessment directory
        quality_dir = os.path.join(workdir, "quality")
        os.makedirs(quality_dir, exist_ok=True)
        
        # Run encapp_quality with the JSON result file
        encapp_metrics = self._run_encapp_quality(
            json_result_file,  # Pass the JSON result file as positional argument
            os.path.dirname(reference_file),  # Reference directory
            quality_dir
        )
        
        if encapp_metrics:
            self.logger.info(f"Quality assessment completed: {encapp_metrics}")
            return encapp_metrics
        
        # Fallback to basic quality assessment
        self.logger.info("Falling back to basic quality assessment")
        try:
            # Find the encoded video file from the JSON result
            with open(json_result_file, 'r') as f:
                json_data = json.load(f)
            
            encoded_file = json_data.get("encodedfile")
            if not encoded_file or not os.path.exists(encoded_file):
                self.logger.warning(f"Encoded video file not found: {encoded_file}")
                return {}
            
            metrics = self.assess_quality(reference_file, encoded_file, quality_dir)
            return {
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "vmaf": metrics.vmaf,
                "bitrate": metrics.bitrate,
                "file_size": metrics.file_size,
                "duration": metrics.duration,
                "resolution": f"{metrics.resolution[0]}x{metrics.resolution[1]}" if metrics.resolution else None,
                "framerate": metrics.framerate,
                "codec": metrics.codec
            }
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {}
    
    def batch_assess_quality(self, test_results: List[Dict[str, Any]], 
                           reference_dir: str, output_dir: str) -> List[QualityMetrics]:
        """
        Batch quality assessment for multiple test results
        
        Args:
            test_results: List of test result dictionaries
            reference_dir: Directory containing reference videos
            output_dir: Directory for quality assessment outputs
            
        Returns:
            List of QualityMetrics objects
        """
        quality_results = []
        
        for i, test_result in enumerate(test_results):
            try:
                # Find reference video
                test_name = test_result.get("test_name", f"test_{i}")
                reference_video = self._find_reference_video(test_name, reference_dir)
                
                if not reference_video:
                    self.logger.warning(f"No reference video found for {test_name}")
                    continue
                
                # Find encoded video
                encoded_video = test_result.get("output_file")
                if not encoded_video or not os.path.exists(encoded_video):
                    self.logger.warning(f"No encoded video found for {test_name}")
                    continue
                
                # Assess quality
                test_output_dir = os.path.join(output_dir, f"quality_{test_name}")
                metrics = self.assess_quality(reference_video, encoded_video, test_output_dir)
                
                # Add test metadata
                metrics.encoder = test_result.get("encoder")
                metrics.additional_metrics["test_name"] = test_name
                metrics.additional_metrics["device_serial"] = test_result.get("device_serial")
                
                quality_results.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Quality assessment failed for test {i}: {e}")
                continue
        
        return quality_results
    
    def batch_assess_quality_with_encapp(self, test_result_files: List[str], 
                                       reference_dir: str, output_dir: str) -> Optional[Dict[str, Any]]:
        """
        Run encapp_quality on multiple test result files with enhanced options
        
        Args:
            test_result_files: List of test result JSON files (positional arguments)
            reference_dir: Directory containing reference videos
            output_dir: Directory for quality assessment outputs
            
        Returns:
            Dictionary with aggregated quality metrics
        """
        try:
            # Look for encapp_quality.py in the lib/encapp directory
            encapp_quality_path = None
            possible_paths = [
                "lib/encapp/scripts/encapp_quality.py",
                "../lib/encapp/scripts/encapp_quality.py",
                "encapp_quality.py"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    encapp_quality_path = path
                    break
            
            if not encapp_quality_path:
                self.logger.debug("encapp_quality.py not found, skipping batch assessment")
                return None
            
            # Calculate max parallel processes (number of cores - 1)
            import multiprocessing
            max_parallel = max(1, multiprocessing.cpu_count() - 1)
            
            # Run encapp_quality on all test result files with enhanced options
            quality_output = os.path.join(output_dir, "batch_quality.csv")
            cmd = [
                "python3", encapp_quality_path,
                *test_result_files,  # All test result files
                "--output", quality_output,
                "--media", reference_dir,
                "--header",  # Add header to CSV data
                "--keep-quality-files",  # Keep intermediate results for time series plotting
                "--ignore-timing",  # Ignore timing information for frame-by-frame comparison
                "--max-parallel", str(max_parallel)  # Use cores - 1 for parallel processing
            ]
            
            self.logger.info(f"Running batch encapp_quality with {max_parallel} parallel processes on {len(test_result_files)} files")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse CSV results
            if os.path.exists(quality_output):
                return self._parse_batch_quality_csv(quality_output)
            
        except Exception as e:
            self.logger.warning(f"Batch encapp_quality failed: {e}")
        
        return None
    
    def _parse_batch_quality_csv(self, csv_file: str) -> Optional[Dict[str, Any]]:
        """Parse batch quality CSV file from encapp_quality"""
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_file)
            if df.empty:
                return None
            
            # Aggregate metrics across all tests
            metrics = {}
            
            # Calculate averages for numeric columns
            numeric_columns = ['psnr_avg', 'ssim', 'vmaf', 'bitrate', 'file_size', 'duration']
            for col in numeric_columns:
                if col in df.columns:
                    metrics[col] = float(df[col].mean())
            
            # Add additional statistics
            if 'vmaf' in df.columns:
                metrics['vmaf_min'] = float(df['vmaf'].min())
                metrics['vmaf_max'] = float(df['vmaf'].max())
                metrics['vmaf_std'] = float(df['vmaf'].std())
            
            if 'psnr_avg' in df.columns:
                metrics['psnr_min'] = float(df['psnr_avg'].min())
                metrics['psnr_max'] = float(df['psnr_avg'].max())
                metrics['psnr_std'] = float(df['psnr_avg'].std())
            
            # Add test count
            metrics['test_count'] = len(df)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to parse batch quality CSV: {e}")
            return None
    
    def _find_reference_video(self, test_name: str, reference_dir: str) -> Optional[str]:
        """Find reference video for a test"""
        reference_dir = Path(reference_dir)
        
        # Try different naming patterns
        patterns = [
            f"{test_name}_reference.mp4",
            f"{test_name}.mp4",
            "reference.mp4",
            "input.mp4"
        ]
        
        for pattern in patterns:
            ref_path = reference_dir / pattern
            if ref_path.exists():
                return str(ref_path)
        
        # Look for any video file in the directory
        video_files = list(reference_dir.glob("*.mp4")) + list(reference_dir.glob("*.mkv"))
        if video_files:
            return str(video_files[0])
        
        return None
    
    def generate_quality_report(self, quality_results: List[QualityMetrics], 
                              output_file: str) -> Dict[str, Any]:
        """Generate a quality assessment report"""
        report = {
            "summary": {
                "total_tests": len(quality_results),
                "successful_assessments": len([r for r in quality_results if r.psnr is not None]),
                "average_psnr": None,
                "average_ssim": None,
                "average_vmaf": None
            },
            "detailed_results": []
        }
        
        # Calculate averages
        psnr_values = [r.psnr for r in quality_results if r.psnr is not None]
        ssim_values = [r.ssim for r in quality_results if r.ssim is not None]
        vmaf_values = [r.vmaf for r in quality_results if r.vmaf is not None]
        
        if psnr_values:
            report["summary"]["average_psnr"] = sum(psnr_values) / len(psnr_values)
        if ssim_values:
            report["summary"]["average_ssim"] = sum(ssim_values) / len(ssim_values)
        if vmaf_values:
            report["summary"]["average_vmaf"] = sum(vmaf_values) / len(vmaf_values)
        
        # Add detailed results
        for result in quality_results:
            report["detailed_results"].append({
                "psnr": result.psnr,
                "ssim": result.ssim,
                "vmaf": result.vmaf,
                "bitrate": result.bitrate,
                "file_size": result.file_size,
                "resolution": result.resolution,
                "framerate": result.framerate,
                "codec": result.codec,
                "encoder": result.encoder,
                "additional_metrics": result.additional_metrics
            })
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


if __name__ == "__main__":
    # Example usage
    quality_assessor = QualityAssessment(debug=True)
    
    # Example quality assessment
    reference = "reference_video.mp4"
    encoded = "encoded_video.mp4"
    
    if os.path.exists(reference) and os.path.exists(encoded):
        metrics = quality_assessor.assess_quality(reference, encoded)
        print(f"Quality metrics: PSNR={metrics.psnr}, SSIM={metrics.ssim}, VMAF={metrics.vmaf}")
    else:
        print("Example videos not found, skipping quality assessment")
