#!/usr/bin/env python3

"""
BD-Rate Utilities for AVA

This module provides utilities for integrating BD-Rate calculations with AVA's test results
and report generation system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json

try:
    from .bd_rate import BDRateCalculator, BDRateResult, compare_codecs_bd_rate, format_bd_rate_result
except ImportError:
    from bd_rate import BDRateCalculator, BDRateResult, compare_codecs_bd_rate, format_bd_rate_result

class AVABDRateAnalyzer:
    """BD-Rate analysis for AVA test results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculator = BDRateCalculator()
    
    def extract_rd_curves_from_csv(self, csv_path: str, 
                                  quality_metric: str = 'psnr') -> Dict[str, pd.DataFrame]:
        """
        Extract rate-distortion curves from AVA CSV files
        
        Args:
            csv_path: Path to CSV file with test results
            quality_metric: Quality metric to use ('psnr', 'ssim', 'vmaf')
            
        Returns:
            Dictionary mapping codec names to RD curve DataFrames
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {csv_path}: {e}")
            return {}
        
        # Map quality metric names
        quality_columns = {
            'psnr': ['psnr', 'psnr_mean', 'psnr_avg'],
            'ssim': ['ssim', 'ssim_mean', 'ssim_avg'], 
            'vmaf': ['vmaf', 'vmaf_mean', 'vmaf_avg']
        }
        
        # Find the correct quality column
        quality_col = None
        if quality_metric in quality_columns:
            for col_pattern in quality_columns[quality_metric]:
                matching_cols = [col for col in df.columns if col_pattern in col.lower()]
                if matching_cols:
                    quality_col = matching_cols[0]
                    break
        
        if quality_col is None:
            self.logger.error(f"Could not find quality column for metric '{quality_metric}'")
            return {}
        
        # Find bitrate column
        bitrate_col = None
        bitrate_candidates = ['calculated_bitrate_bps', 'bitrate_bps', 'bitrate', 'meanbitrate']
        for col in bitrate_candidates:
            if col in df.columns:
                bitrate_col = col
                break
        
        if bitrate_col is None:
            self.logger.error("Could not find bitrate column")
            return {}
        
        # Group by codec and extract RD curves
        rd_curves = {}
        
        if 'codec' not in df.columns:
            self.logger.error("No 'codec' column found in CSV")
            return {}
        
        for codec in df['codec'].unique():
            codec_data = df[df['codec'] == codec].copy()
            
            # Filter valid data points
            valid_data = codec_data.dropna(subset=[bitrate_col, quality_col])
            valid_data = valid_data[valid_data[quality_col] > 0]
            
            if len(valid_data) < 2:
                self.logger.warning(f"Insufficient data points for codec {codec}")
                continue
            
            # Sort by quality for proper RD curve
            valid_data = valid_data.sort_values(quality_col)
            
            rd_curves[codec] = valid_data[[bitrate_col, quality_col, 'codec']].copy()
            rd_curves[codec].columns = ['bitrate', 'quality', 'codec']
        
        return rd_curves
    
    def calculate_bd_rate_from_csv(self, csv_path: str, 
                                 codec1: str, 
                                 codec2: str,
                                 quality_metric: str = 'psnr') -> Optional[BDRateResult]:
        """
        Calculate BD-Rate between two codecs from CSV file
        
        Args:
            csv_path: Path to CSV file with test results
            codec1: First codec name
            codec2: Second codec name
            quality_metric: Quality metric to use
            
        Returns:
            BDRateResult or None if calculation fails
        """
        rd_curves = self.extract_rd_curves_from_csv(csv_path, quality_metric)
        
        if codec1 not in rd_curves or codec2 not in rd_curves:
            self.logger.error(f"Codecs {codec1} or {codec2} not found in data")
            return None
        
        try:
            # The RD curves already have the correct column names ('bitrate', 'quality', 'codec')
            # Map the quality metric to the generic 'quality' column name
            return self.calculator.calculate_bd_rate(
                rd_curves[codec1], 
                rd_curves[codec2], 
                'quality',  # Use generic quality name since data is already prepared
                'bitrate',  # Use generic bitrate name
                'quality'   # Use generic quality name
            )
        except Exception as e:
            self.logger.error(f"BD-Rate calculation failed: {e}")
            return None
    
    def compare_all_codecs_from_csv(self, csv_path: str, 
                                  quality_metric: str = 'psnr') -> Dict[str, BDRateResult]:
        """
        Compare all codecs pairwise from CSV file
        
        Args:
            csv_path: Path to CSV file with test results
            quality_metric: Quality metric to use
            
        Returns:
            Dictionary mapping codec pairs to BD-Rate results
        """
        rd_curves = self.extract_rd_curves_from_csv(csv_path, quality_metric)
        
        if len(rd_curves) < 2:
            self.logger.warning("Need at least 2 codecs for comparison")
            return {}
        
        results = {}
        codecs = list(rd_curves.keys())
        
        # Compare all pairs
        for i, codec1 in enumerate(codecs):
            for codec2 in codecs[i+1:]:
                try:
                    result = self.calculator.calculate_bd_rate(
                        rd_curves[codec1], 
                        rd_curves[codec2], 
                        'quality',  # Use generic quality name since data is already prepared
                        'bitrate',  # Use generic bitrate name
                        'quality'   # Use generic quality name
                    )
                    results[f"{codec1}_vs_{codec2}"] = result
                except Exception as e:
                    self.logger.warning(f"Failed to compare {codec1} vs {codec2}: {e}")
        
        return results
    
    def compare_codecs_against_reference(self, csv_path: str, 
                                       reference_codec: str,
                                       quality_metric: str = 'psnr') -> Dict[str, BDRateResult]:
        """
        Compare all codecs against a reference codec
        
        Args:
            csv_path: Path to CSV file with test results
            reference_codec: Name of the reference codec
            quality_metric: Quality metric to use
            
        Returns:
            Dictionary mapping codec names to BD-Rate results (vs reference)
        """
        rd_curves = self.extract_rd_curves_from_csv(csv_path, quality_metric)
        
        if len(rd_curves) < 2:
            self.logger.warning("Need at least 2 codecs for comparison")
            return {}
        
        if reference_codec not in rd_curves:
            self.logger.error(f"Reference codec '{reference_codec}' not found in data")
            return {}
        
        results = {}
        reference_data = rd_curves[reference_codec]
        
        # Compare each codec against the reference
        for codec, data in rd_curves.items():
            if codec == reference_codec:
                continue  # Skip self-comparison
                
            try:
                result = self.calculator.calculate_bd_rate(
                    data,  # Test codec
                    reference_data,  # Reference codec
                    'quality',  # Use generic quality name since data is already prepared
                    'bitrate',  # Use generic bitrate name
                    'quality'   # Use generic quality name
                )
                results[codec] = result
            except Exception as e:
                self.logger.warning(f"Failed to compare {codec} vs {reference_codec}: {e}")
        
        return results
    
    def get_available_codecs(self, csv_path: str) -> List[str]:
        """
        Get list of available codecs from CSV file
        
        Args:
            csv_path: Path to CSV file with test results
            
        Returns:
            List of codec names
        """
        try:
            df = pd.read_csv(csv_path)
            if 'codec' not in df.columns:
                return []
            return sorted(df['codec'].unique().tolist())
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {csv_path}: {e}")
            return []
    
    def generate_bd_rate_report(self, csv_path: str, 
                              output_path: str = None,
                              quality_metrics: List[str] = None) -> str:
        """
        Generate comprehensive BD-Rate comparison report
        
        Args:
            csv_path: Path to CSV file with test results
            output_path: Optional output path for report file
            quality_metrics: List of quality metrics to analyze
            
        Returns:
            Report content as string
        """
        if quality_metrics is None:
            quality_metrics = ['psnr', 'ssim', 'vmaf']
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Bjøntegaard-Delta (BD-Rate) Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Source: {csv_path}")
        report_lines.append("")
        
        for metric in quality_metrics:
            report_lines.append(f"Quality Metric: {metric.upper()}")
            report_lines.append("-" * 40)
            
            try:
                results = self.compare_all_codecs_from_csv(csv_path, metric)
                
                if not results:
                    report_lines.append("No valid comparisons found for this metric.")
                    report_lines.append("")
                    continue
                
                for pair_name, result in results.items():
                    report_lines.append(f"\nComparison: {pair_name}")
                    report_lines.append(format_bd_rate_result(result))
                    report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"Error analyzing {metric}: {e}")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"BD-Rate report saved to {output_path}")
        
        return report_content

def find_quality_csv_files(workdir: str) -> List[str]:
    """
    Find quality CSV files in work directory
    
    Args:
        workdir: Working directory to search
        
    Returns:
        List of paths to quality CSV files
    """
    workdir_path = Path(workdir)
    csv_files = []
    
    # Look for common quality CSV file patterns
    patterns = [
        "**/quality_*.csv",
        "**/*_quality.csv", 
        "**/quality.csv",
        "**/all_quality_data.csv"
    ]
    
    for pattern in patterns:
        csv_files.extend(workdir_path.glob(pattern))
    
    return [str(f) for f in csv_files]

def analyze_workdir_bd_rate(workdir: str, 
                          output_dir: str = None,
                          quality_metrics: List[str] = None) -> Dict[str, Any]:
    """
    Analyze BD-Rate for all quality CSV files in work directory
    
    Args:
        workdir: Working directory containing test results
        output_dir: Optional output directory for reports
        quality_metrics: List of quality metrics to analyze
        
    Returns:
        Dictionary with analysis results
    """
    if quality_metrics is None:
        quality_metrics = ['psnr', 'ssim', 'vmaf']
    
    analyzer = AVABDRateAnalyzer()
    csv_files = find_quality_csv_files(workdir)
    
    if not csv_files:
        logging.getLogger(__name__).warning(f"No quality CSV files found in {workdir}")
        return {}
    
    results = {}
    
    for csv_file in csv_files:
        file_name = Path(csv_file).stem
        results[file_name] = {}
        
        for metric in quality_metrics:
            try:
                bd_results = analyzer.compare_all_codecs_from_csv(csv_file, metric)
                results[file_name][metric] = bd_results
                
                # Generate report if output directory specified
                if output_dir:
                    output_path = Path(output_dir) / f"{file_name}_bd_rate_{metric}.txt"
                    analyzer.generate_bd_rate_report(csv_file, str(output_path), [metric])
                    
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to analyze {csv_file} for {metric}: {e}")
                results[file_name][metric] = {}
    
    return results
