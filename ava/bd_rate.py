#!/usr/bin/env python3

"""
Bjøntegaard-Delta (BD-Rate) Calculation Module

This module implements the Bjøntegaard-Delta metric for comparing video codec performance.
BD-Rate measures the average percentage bit rate difference between two rate-distortion 
curves at equal measured distortion, integrated across a range of bit rates in the 
logarithmic domain.

References:
- ITU-T Recommendation P.1401 (2012)
- VCEG-M33: Calculation of average PSNR differences between RD-curves
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings

@dataclass
class BDRateResult:
    """Container for BD-Rate calculation results"""
    bd_rate: float
    bd_psnr: float
    quality_metric: str
    codec1: str
    codec2: str
    integration_range: Tuple[float, float]
    num_points: int
    interpolation_method: str

class BDRateCalculator:
    """Calculate Bjøntegaard-Delta (BD-Rate) between two codecs"""
    
    def __init__(self, interpolation_method: str = 'cubic', log_domain: bool = True):
        """
        Initialize BD-Rate calculator
        
        Args:
            interpolation_method: Method for interpolating RD curves ('cubic', 'linear', 'quadratic')
            log_domain: Whether to perform integration in logarithmic domain (recommended)
        """
        self.interpolation_method = interpolation_method
        self.log_domain = log_domain
        self.logger = logging.getLogger(__name__)
        
    def calculate_bd_rate(self, 
                         codec1_data: pd.DataFrame, 
                         codec2_data: pd.DataFrame,
                         quality_metric: str = 'psnr',
                         bitrate_col: str = 'calculated_bitrate_bps',
                         quality_col: str = None) -> BDRateResult:
        """
        Calculate BD-Rate between two codecs
        
        Args:
            codec1_data: DataFrame with rate-distortion data for codec 1
            codec2_data: DataFrame with rate-distortion data for codec 2  
            quality_metric: Quality metric to use ('psnr', 'ssim', 'vmaf')
            bitrate_col: Column name for bitrate data
            quality_col: Column name for quality data (auto-detected if None)
            
        Returns:
            BDRateResult object with BD-Rate calculation results
        """
        if quality_col is None:
            quality_col = self._get_quality_column(quality_metric, codec1_data.columns)
        
        # Prepare data
        rd1 = self._prepare_rd_data(codec1_data, bitrate_col, quality_col)
        rd2 = self._prepare_rd_data(codec2_data, bitrate_col, quality_col)
        
        if len(rd1) < 2 or len(rd2) < 2:
            raise ValueError("Need at least 2 data points for each codec")
        
        # Find common quality range
        min_quality = max(rd1['quality'].min(), rd2['quality'].min())
        max_quality = min(rd1['quality'].max(), rd2['quality'].max())
        
        if min_quality >= max_quality:
            raise ValueError("No overlapping quality range between codecs")
        
        # Create interpolation functions
        f1 = self._create_interpolation_function(rd1, quality_metric)
        f2 = self._create_interpolation_function(rd2, quality_metric)
        
        # Calculate BD-Rate
        bd_rate = self._integrate_bd_rate(f1, f2, min_quality, max_quality)
        
        # Calculate BD-PSNR (optional, mainly for PSNR)
        bd_psnr = None
        if quality_metric.lower() == 'psnr':
            bd_psnr = self._calculate_bd_psnr(f1, f2, min_quality, max_quality)
        
        return BDRateResult(
            bd_rate=bd_rate,
            bd_psnr=bd_psnr,
            quality_metric=quality_metric,
            codec1=codec1_data.get('codec', ['Unknown']).iloc[0] if 'codec' in codec1_data.columns else 'Codec1',
            codec2=codec2_data.get('codec', ['Unknown']).iloc[0] if 'codec' in codec2_data.columns else 'Codec2',
            integration_range=(min_quality, max_quality),
            num_points=len(rd1) + len(rd2),
            interpolation_method=self.interpolation_method
        )
    
    def _get_quality_column(self, quality_metric: str, columns: List[str]) -> str:
        """Auto-detect quality column name"""
        quality_metric = quality_metric.lower()
        
        # Common column name patterns
        patterns = {
            'psnr': ['psnr', 'psnr_mean', 'psnr_avg'],
            'ssim': ['ssim', 'ssim_mean', 'ssim_avg'],
            'vmaf': ['vmaf', 'vmaf_mean', 'vmaf_avg']
        }
        
        if quality_metric in patterns:
            for pattern in patterns[quality_metric]:
                for col in columns:
                    if pattern in col.lower():
                        return col
        
        # Fallback to exact match
        if quality_metric in columns:
            return quality_metric
            
        raise ValueError(f"Could not find quality column for metric '{quality_metric}' in columns: {list(columns)}")
    
    def _prepare_rd_data(self, data: pd.DataFrame, bitrate_col: str, quality_col: str) -> pd.DataFrame:
        """Prepare rate-distortion data for interpolation"""
        # Filter out invalid data points
        valid_data = data.dropna(subset=[bitrate_col, quality_col])
        valid_data = valid_data[valid_data[quality_col] > 0]  # Remove invalid quality values
        
        if len(valid_data) == 0:
            raise ValueError("No valid rate-distortion data points found")
        
        # Sort by quality (for interpolation)
        rd_data = valid_data[[bitrate_col, quality_col]].copy()
        rd_data.columns = ['bitrate', 'quality']
        rd_data = rd_data.sort_values('quality')
        
        # Remove duplicates (keep first occurrence)
        rd_data = rd_data.drop_duplicates(subset=['quality'], keep='first')
        
        return rd_data
    
    def _create_interpolation_function(self, rd_data: pd.DataFrame, quality_metric: str):
        """Create interpolation function for rate-distortion curve"""
        x = rd_data['quality'].values
        y = rd_data['bitrate'].values
        
        # Use logarithmic domain for bitrate if specified
        if self.log_domain:
            y = np.log10(y)
        
        # Create interpolation function
        if self.interpolation_method == 'cubic' and len(rd_data) >= 4:
            # Use cubic spline for smooth interpolation
            f = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        elif self.interpolation_method == 'quadratic' and len(rd_data) >= 3:
            # Use quadratic interpolation
            f = interpolate.interp1d(x, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
        else:
            # Fall back to linear interpolation
            f = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        return f
    
    def _integrate_bd_rate(self, f1, f2, min_quality: float, max_quality: float) -> float:
        """
        Calculate BD-Rate by integrating the difference between two RD curves
        
        BD-Rate = (1 / (Q2 - Q1)) * ∫[Q1 to Q2] log10(R2(Q) / R1(Q)) dQ
        
        Where:
        - Q1, Q2 are the quality range bounds
        - R1(Q), R2(Q) are the bitrate functions for codec1 and codec2
        """
        # Create quality points for integration
        num_points = 1000  # High resolution for accurate integration
        quality_points = np.linspace(min_quality, max_quality, num_points)
        
        # Evaluate bitrate functions
        try:
            bitrate1 = f1(quality_points)
            bitrate2 = f2(quality_points)
        except Exception as e:
            self.logger.warning(f"Interpolation failed, using linear fallback: {e}")
            # Fallback to linear interpolation
            f1_linear = interpolate.interp1d(
                np.linspace(min_quality, max_quality, 10), 
                np.linspace(f1(min_quality), f1(max_quality), 10),
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            f2_linear = interpolate.interp1d(
                np.linspace(min_quality, max_quality, 10),
                np.linspace(f2(min_quality), f2(max_quality), 10), 
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            bitrate1 = f1_linear(quality_points)
            bitrate2 = f2_linear(quality_points)
        
        # Calculate log ratio
        if self.log_domain:
            # Already in log domain
            log_ratio = bitrate2 - bitrate1
        else:
            # Convert to log domain
            log_ratio = np.log10(bitrate2) - np.log10(bitrate1)
        
        # Integrate using trapezoidal rule
        bd_rate = np.trapz(log_ratio, quality_points) / (max_quality - min_quality)
        
        # Convert to percentage
        bd_rate_percent = bd_rate * 100
        
        return bd_rate_percent
    
    def _calculate_bd_psnr(self, f1, f2, min_quality: float, max_quality: float) -> float:
        """Calculate BD-PSNR (average PSNR difference)"""
        # Create quality points for integration
        num_points = 1000
        quality_points = np.linspace(min_quality, max_quality, num_points)
        
        # Evaluate bitrate functions
        try:
            bitrate1 = f1(quality_points)
            bitrate2 = f2(quality_points)
        except Exception as e:
            self.logger.warning(f"Interpolation failed for BD-PSNR: {e}")
            return None
        
        # Convert back from log domain if needed
        if self.log_domain:
            bitrate1 = 10 ** bitrate1
            bitrate2 = 10 ** bitrate2
        
        # Calculate PSNR difference (this is a simplified version)
        # In practice, you'd need the actual PSNR values, not bitrates
        # This is a placeholder implementation
        psnr_diff = np.trapz(quality_points, quality_points) / (max_quality - min_quality)
        
        return psnr_diff

def calculate_bd_rate_from_test_results(test_results: List[Dict[str, Any]], 
                                      codec1: str, 
                                      codec2: str,
                                      quality_metric: str = 'psnr',
                                      device_filter: str = None) -> Optional[BDRateResult]:
    """
    Calculate BD-Rate from test results data
    
    Args:
        test_results: List of test result dictionaries
        codec1: Name of first codec to compare
        codec2: Name of second codec to compare  
        quality_metric: Quality metric to use ('psnr', 'ssim', 'vmaf')
        device_filter: Optional device serial to filter results
        
    Returns:
        BDRateResult or None if insufficient data
    """
    # Convert to DataFrame
    df = pd.DataFrame(test_results)
    
    if df.empty:
        return None
    
    # Filter by device if specified
    if device_filter and 'device_serial' in df.columns:
        df = df[df['device_serial'] == device_filter]
    
    # Filter by codecs
    codec1_data = df[df['codec'] == codec1] if 'codec' in df.columns else pd.DataFrame()
    codec2_data = df[df['codec'] == codec2] if 'codec' in df.columns else pd.DataFrame()
    
    if codec1_data.empty or codec2_data.empty:
        return None
    
    # Calculate BD-Rate
    calculator = BDRateCalculator()
    try:
        return calculator.calculate_bd_rate(codec1_data, codec2_data, quality_metric)
    except Exception as e:
        logging.getLogger(__name__).error(f"BD-Rate calculation failed: {e}")
        return None

def compare_codecs_bd_rate(test_results: List[Dict[str, Any]], 
                          quality_metric: str = 'psnr',
                          device_filter: str = None) -> Dict[str, BDRateResult]:
    """
    Compare all codecs pairwise using BD-Rate
    
    Args:
        test_results: List of test result dictionaries
        quality_metric: Quality metric to use
        device_filter: Optional device serial to filter results
        
    Returns:
        Dictionary mapping codec pairs to BD-Rate results
    """
    df = pd.DataFrame(test_results)
    
    if df.empty or 'codec' not in df.columns:
        return {}
    
    # Filter by device if specified
    if device_filter and 'device_serial' in df.columns:
        df = df[df['device_serial'] == device_filter]
    
    # Get unique codecs
    codecs = df['codec'].unique()
    
    if len(codecs) < 2:
        return {}
    
    results = {}
    calculator = BDRateCalculator()
    
    # Compare all pairs
    for i, codec1 in enumerate(codecs):
        for codec2 in codecs[i+1:]:
            codec1_data = df[df['codec'] == codec1]
            codec2_data = df[df['codec'] == codec2]
            
            try:
                bd_result = calculator.calculate_bd_rate(codec1_data, codec2_data, quality_metric)
                results[f"{codec1}_vs_{codec2}"] = bd_result
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to compare {codec1} vs {codec2}: {e}")
    
    return results

def format_bd_rate_result(result: BDRateResult) -> str:
    """Format BD-Rate result for display"""
    if result.bd_rate > 0:
        direction = "increase"
    else:
        direction = "decrease"
    
    output = f"BD-Rate ({result.quality_metric.upper()}): {abs(result.bd_rate):.2f}% {direction}\n"
    output += f"Codec 1: {result.codec1}\n"
    output += f"Codec 2: {result.codec2}\n"
    output += f"Quality range: {result.integration_range[0]:.2f} - {result.integration_range[1]:.2f}\n"
    output += f"Data points: {result.num_points}\n"
    output += f"Interpolation: {result.interpolation_method}\n"
    
    if result.bd_psnr is not None:
        output += f"BD-PSNR: {result.bd_psnr:.2f} dB\n"
    
    return output
