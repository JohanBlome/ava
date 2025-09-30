"""
AVA Calculator Module

This module contains all calculation functions extracted from ava_report.py
to separate data processing logic from report generation logic.
"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


class MetricCalculator:
    """Handles all metric calculations for AVA reports"""
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the calculator with optional logger"""
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_frame_statistics(self, device_data: pd.DataFrame, metric_column: str) -> tuple:
        """Calculate frame-based statistics with confidence intervals"""
        # Group by frame and calculate statistics
        frame_stats = device_data.groupby('frame').agg({
            metric_column: ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        frame_stats.columns = ['frame', 'mean_value', 'std_value', 'count']
        
        # Convert frame numbers to time (proper 1-second timeline)
        fps = device_data['fps'].iloc[0] if 'fps' in device_data.columns else 30
        time_sec = (frame_stats['frame'] / fps).values
        mean_values = frame_stats['mean_value'].values
        std_values = frame_stats['std_value'].values
        count = frame_stats['count'].values
        
        # Handle NaN standard deviations (when only one data point per frame)
        # For single data points, use a small percentage of the mean as std
        std_values = np.where(np.isnan(std_values), mean_values * 0.05, std_values)
        
        # Calculate confidence interval (95% CI)
        # Use t-distribution for small samples, normal for large samples
        confidence_factor = 1.96  # 95% CI for normal distribution
        upper_bound = mean_values + confidence_factor * std_values
        lower_bound = mean_values - confidence_factor * std_values
        
        # Ensure lower bound is never negative for bitrate
        if 'bitrate' in metric_column.lower():
            lower_bound = np.maximum(lower_bound, 0)
        
        return time_sec, mean_values, upper_bound, lower_bound, count
    
    def calculate_metric_statistics(self, combined_df: pd.DataFrame, metric_column: str, bitrate_column: str = 'calculated_bitrate_bps') -> dict:
        """
        Calculate any metric (e.g. vmaf_mean, psnr, ssim) over multiple sources with confidence intervals.
        
        This function is designed to prepare data for smoother Quality Metric graphs by aggregating
        metrics across multiple sources (devices) and reference files, providing confidence intervals.
        It properly averages over runs with different target bitrates and different reference files.
        
        Args:
            combined_df: DataFrame containing all test results (must include 'reference_file' column)
            metric_column: Column name for the metric to calculate (e.g., 'vmaf_mean', 'psnr', 'ssim')
            bitrate_column: Column name for bitrate (e.g., 'calculated_bitrate_bps', 'bitrate_bps')
            
        Returns:
            Dictionary with structure:
            {
                'codec_name': {
                    'bitrates': array of bitrate values,
                    'mean_values': array of mean metric values (averaged across reference files),
                    'upper_bound': array of upper confidence bound values,
                    'lower_bound': array of lower confidence bound values,
                    'count': array of sample counts,
                    'sources': list of source identifiers (devices and reference files)
                }
            }
            
        Note:
            The function now properly handles different reference files by averaging metrics
            across all reference files and devices at each bitrate point. This ensures that
            interpolation results are more accurate when comparing across different test runs.
            
            Bitrate binning: Similar bitrates (within 5% tolerance) are grouped together
            and averaged, which is especially useful when testing the same target bitrates
            with different reference files that may achieve slightly different actual bitrates.
            
            Confidence intervals are calculated as follows:
            - 1 data point: No confidence interval (upper = lower = mean)
            - 2+ data points: Statistical confidence intervals using t-distribution (small samples) or normal distribution (large samples)
            - All confidence intervals are properly centered around the mean and represent statistical uncertainty
            
        Example:
            # Calculate VMAF statistics across multiple sources and reference files
            vmaf_stats = calculator.calculate_metric_statistics(combined_df, 'vmaf_mean', 'calculated_bitrate_bps')
            
            # Calculate PSNR statistics using target bitrate
            psnr_stats = calculator.calculate_metric_statistics(combined_df, 'psnr', 'bitrate_bps')
            
            # Access data for a specific codec
            if 'h264' in vmaf_stats:
                codec_data = vmaf_stats['h264']
                bitrates = codec_data['bitrates']
                mean_vmaf = codec_data['mean_values']  # Averaged across reference files
                confidence_upper = codec_data['upper_bound']
                confidence_lower = codec_data['lower_bound']
        """
        results = {}
        
        # Validate input columns
        if metric_column not in combined_df.columns:
            self.logger.warning(f"Metric column '{metric_column}' not found in data")
            return results
            
        if bitrate_column not in combined_df.columns:
            self.logger.warning(f"Bitrate column '{bitrate_column}' not found in data")
            return results
        
        # Filter out invalid values for the metric
        valid_data = combined_df.copy()
        if metric_column in ['psnr', 'ssim']:
            # PSNR and SSIM use -1 to indicate invalid values
            valid_data = valid_data[valid_data[metric_column] != -1]
        elif metric_column == 'vmaf_mean':
            # VMAF should be between 0 and 100
            valid_data = valid_data[(valid_data[metric_column] >= 0) & (valid_data[metric_column] <= 100)]
        
        if valid_data.empty:
            self.logger.warning(f"No valid data found for metric '{metric_column}'")
            return results
        
        # Check if reference_file column exists, if not create a dummy one
        if 'reference_file' not in valid_data.columns:
            valid_data['reference_file'] = 'unknown'
        
        # Group by codec and calculate statistics
        for codec in valid_data['codec'].unique():
            codec_data = valid_data[valid_data['codec'] == codec]
            
            # Sort by bitrate for consistent ordering
            codec_data = codec_data.sort_values(bitrate_column)
            
            # Get unique bitrate values and group similar bitrates together
            # Use binning to group similar bitrates (within 5% tolerance)
            all_bitrates = codec_data[bitrate_column].values
            unique_bitrates = sorted(codec_data[bitrate_column].unique())
            
            if len(unique_bitrates) < 2:
                self.logger.warning(f"Not enough bitrate points for codec '{codec}' (need at least 2)")
                continue
            
            # Group similar bitrates together using 5% tolerance
            bitrate_groups = []
            current_group = [unique_bitrates[0]]
            
            for i in range(1, len(unique_bitrates)):
                current_bitrate = unique_bitrates[i]
                group_center = np.mean(current_group)
                
                # If within 5% tolerance, add to current group
                if abs(current_bitrate - group_center) / group_center <= 0.05:
                    current_group.append(current_bitrate)
                else:
                    # Start new group
                    bitrate_groups.append(current_group)
                    current_group = [current_bitrate]
            
            # Add the last group
            bitrate_groups.append(current_group)
            
            # Calculate statistics for each bitrate group, averaging across different reference files
            bitrates = []
            mean_values = []
            upper_bounds = []
            lower_bounds = []
            counts = []
            all_sources = set()
            
            for bitrate_group in bitrate_groups:
                # Find all data points within this bitrate group
                group_center = np.mean(bitrate_group)
                tolerance = group_center * 0.05
                
                bitrate_data = codec_data[
                    (codec_data[bitrate_column] >= group_center - tolerance) &
                    (codec_data[bitrate_column] <= group_center + tolerance)
                ]
                
                if bitrate_data.empty:
                    continue
                
                # Calculate statistics across all sources (devices) and reference files at this bitrate
                metric_values = bitrate_data[metric_column].values
                mean_val = np.mean(metric_values)
                count = len(metric_values)
                
                # Calculate confidence intervals
                if count == 1:
                    # Single data point - no confidence interval
                    upper_bound = mean_val
                    lower_bound = mean_val
                else:
                    # Calculate proper confidence intervals
                    std_val = np.std(metric_values, ddof=1)  # Sample standard deviation
                    
                    if count < 30:
                        # Use t-distribution for small samples
                        from scipy import stats
                        t_val = stats.t.ppf(0.975, count - 1)  # 95% CI
                        margin_error = t_val * std_val / np.sqrt(count)
                    else:
                        # Use normal distribution for large samples
                        # Calculate confidence interval using seaborn's bootstrap approach
                        if count >= 10:
                            # Resample the data multiple times and calculate mean
                            bootstrap_means = []
                            np.random.seed(42)  # For reproducibility
                            for _ in range(1000):
                                bootstrap_sample = np.random.choice(metric_values, size=count, replace=True)
                                bootstrap_means.append(np.mean(bootstrap_sample))
                            
                            # Calculate 95% confidence interval from bootstrap distribution
                            bootstrap_means = np.array(bootstrap_means)
                            margin_error = np.percentile(bootstrap_means, 97.5) - np.percentile(bootstrap_means, 2.5)
                            margin_error /= 2  # Convert to ±margin
                        else:
                            # For very small samples, use simple approximation
                            data_range = np.max(metric_values) - np.min(metric_values)
                            margin_error = data_range * 0.25  # Conservative estimate
                    
                    # Ensure bounds are reasonable for the metric while maintaining symmetry
                    if metric_column in ['psnr', 'ssim', 'vmaf_mean']:
                        if metric_column == 'vmaf_mean':
                            # Keep confidence interval symmetric around mean
                            actual_half_range = min(mean_val, 100 - mean_val)
                            symmetric_range = min(margin_error, actual_half_range)  # Don't exceed boundaries
                            upper_bound = mean_val + symmetric_range
                            lower_bound = mean_val - symmetric_range
                        elif metric_column == 'psnr':
                            # Keep confidence interval symmetric around mean
                            actual_half_range = mean_val  # PSNR can't be negative
                            symmetric_range = min(margin_error, actual_half_range)
                            upper_bound = mean_val + symmetric_range
                            lower_bound = mean_val - symmetric_range
                        elif metric_column == 'ssim':
                            # Keep confidence interval symmetric around mean
                            actual_half_range = min(mean_val, 1.0 - mean_val)
                            symmetric_range = min(margin_error, actual_half_range)
                            upper_bound = mean_val + symmetric_range
                            lower_bound = mean_val - symmetric_range
                    else:
                        # For other metrics, use standard confidence intervals
                        upper_bound = mean_val + margin_error
                        lower_bound = mean_val - margin_error
                
                # Store results
                bitrates.append(group_center)
                mean_values.append(mean_val)
                upper_bounds.append(upper_bound)
                lower_bounds.append(lower_bound)
                counts.append(count)
                
                # Collect source information
                for _, row in bitrate_data.iterrows():
                    source_id = f"{row.get('device', 'unknown')}_{row.get('reference_file', 'unknown')}"
                    all_sources.add(source_id)
            
            if bitrates:  # Only add if we have valid data
                results[codec] = {
                    'bitrates': np.array(bitrates),
                    'mean_values': np.array(mean_values),
                    'upper_bound': np.array(upper_bounds),
                    'lower_bound': np.array(lower_bounds),
                    'count': np.array(counts),
                    'sources': list(all_sources)
                }
        
        return results
    
    def calculate_si_ti(self, video_file: str) -> Tuple[float, float]:
        """Calculate SI/TI values for a video file using FFmpeg"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", prefix="siti.", delete=False) as tfo:
                tf = tfo.name
            
            # Use FFmpeg to calculate SI/TI
            cmd = [
                "ffmpeg", "-hide_banner", "-y",
                "-i", video_file,
                "-filter_complex", "siti=print_summary=1",
                "-f", "null", "-"
            ]
            
            with open(tf, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=f, check=True)
            
            # Parse the output
            si_avg = -1.0
            ti_avg = -1.0
            
            with open(tf, "r") as f:
                lines = f.readlines()
                spatial = True
                for line in lines:
                    line = line.lower().strip()
                    if "spatial" in line:
                        spatial = True
                        continue
                    if "temporal" in line:
                        spatial = False
                        continue
                    if "average:" in line:
                        try:
                            value = float(line.split(":")[-1].strip())
                            if spatial:
                                si_avg = value
                            else:
                                ti_avg = value
                        except (ValueError, IndexError):
                            continue
            
            # Clean up temp file
            os.unlink(tf)
            
            return si_avg, ti_avg
            
        except Exception as e:
            self.logger.error(f"Failed to calculate SI/TI for {video_file}: {e}")
            return -1.0, -1.0


class BitrateDataProcessor:
    """Handles bitrate data processing and preparation for visualization"""
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the processor with optional logger"""
        self.logger = logger or logging.getLogger(__name__)
    
    def prepare_bitrate_data(self, combined_df: pd.DataFrame, bitrate_col: str, bitrate_label: str, 
                           metric_calculator: MetricCalculator, 
                           get_device_color_func, get_codec_type_func, 
                           get_codec_line_style_func, is_custom_labeled_codec_func) -> Dict[str, Any]:
        """Prepare data for a specific bitrate mode"""
        if bitrate_col not in combined_df.columns:
            return {'traces': [], 'bitrate_label': bitrate_label}
        
        traces = []
        
        # VMAF plot data
        if 'vmaf_mean' in combined_df.columns:
            # Calculate VMAF statistics across multiple sources
            vmaf_stats = metric_calculator.calculate_metric_statistics(combined_df, 'vmaf_mean', bitrate_col)
            
            for codec, stats_data in vmaf_stats.items():
                # Get consistent color and line style
                color = get_device_color_func(codec) if is_custom_labeled_codec_func(codec) else get_device_color_func(codec)
                codec_type = get_codec_type_func(codec)
                line_style = get_codec_line_style_func(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace
                traces.append({
                    'type': 'scatter',
                    'x': bitrates_kbps.tolist(),
                    'y': stats_data['mean_values'].tolist(),
                    'mode': 'markers+lines',
                    'name': trace_name,
                    'line': {'color': color, 'dash': line_style, 'width': 3},
                    'marker': {'size': 6},
                    'legendgroup': codec,
                    'showlegend': True,
                    'row': 1,
                    'col': 1,
                    'hovertemplate': f"<b>{codec}</b><br>" +
                                   f"Bitrate: %{{x:.1f}} kbps<br>" +
                                   f"VMAF: %{{y:.2f}}<br>" +
                                   f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                   f"Sources: {len(stats_data['sources'])}<br>" +
                                   "<extra></extra>"
                })
                
                # Add confidence interval trace only if we have meaningful confidence intervals
                if not np.allclose(stats_data['upper_bound'], stats_data['lower_bound']):
                    # Convert hex color to rgba with transparency
                    if color.startswith('#'):
                        # Convert hex to rgb
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        fillcolor = f'rgba({r}, {g}, {b}, 0.2)'
                    else:
                        # Fallback to gray if color format is unexpected
                        fillcolor = 'rgba(128, 128, 128, 0.2)'
                    
                    traces.append({
                        'type': 'scatter',
                        'x': bitrates_kbps.tolist() + bitrates_kbps[::-1].tolist(),
                        'y': stats_data['upper_bound'].tolist() + stats_data['lower_bound'][::-1].tolist(),
                        'fill': 'tonexty',
                        'fillcolor': fillcolor,
                        'line': {'color': 'rgba(255,255,255,0)'},
                        'legendgroup': codec,
                        'showlegend': False,
                        'row': 1,
                        'col': 1,
                        'hoverinfo': 'skip'
                    })
        
        # PSNR plot data
        if 'psnr' in combined_df.columns:
            # Calculate PSNR statistics across multiple sources
            psnr_stats = metric_calculator.calculate_metric_statistics(combined_df, 'psnr', bitrate_col)
            
            for codec, stats_data in psnr_stats.items():
                # Get consistent color and line style
                color = get_device_color_func(codec) if is_custom_labeled_codec_func(codec) else get_device_color_func(codec)
                codec_type = get_codec_type_func(codec)
                line_style = get_codec_line_style_func(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace
                traces.append({
                    'type': 'scatter',
                    'x': bitrates_kbps.tolist(),
                    'y': stats_data['mean_values'].tolist(),
                    'mode': 'markers+lines',
                    'name': trace_name,
                    'line': {'color': color, 'dash': line_style, 'width': 3},
                    'marker': {'size': 6},
                    'legendgroup': codec,
                    'showlegend': False,
                    'row': 1,
                    'col': 2,
                    'hovertemplate': f"<b>{codec}</b><br>" +
                                   f"Bitrate: %{{x:.1f}} kbps<br>" +
                                   f"PSNR: %{{y:.2f}} dB<br>" +
                                   f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                   f"Sources: {len(stats_data['sources'])}<br>" +
                                   "<extra></extra>"
                })
                
                # Add confidence interval trace only if we have meaningful confidence intervals
                if not np.allclose(stats_data['upper_bound'], stats_data['lower_bound']):
                    # Convert hex color to rgba with transparency
                    if color.startswith('#'):
                        # Convert hex to rgb
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        fillcolor = f'rgba({r}, {g}, {b}, 0.2)'
                    else:
                        # Fallback to gray if color format is unexpected
                        fillcolor = 'rgba(128, 128, 128, 0.2)'
                    
                    traces.append({
                        'type': 'scatter',
                        'x': bitrates_kbps.tolist() + bitrates_kbps[::-1].tolist(),
                        'y': stats_data['upper_bound'].tolist() + stats_data['lower_bound'][::-1].tolist(),
                        'fill': 'tonexty',
                        'fillcolor': fillcolor,
                        'line': {'color': 'rgba(255,255,255,0)'},
                        'legendgroup': codec,
                        'showlegend': False,
                        'row': 1,
                        'col': 2,
                        'hoverinfo': 'skip'
                    })
        
        # SSIM plot data
        if 'ssim' in combined_df.columns:
            # Calculate SSIM statistics across multiple sources
            ssim_stats = metric_calculator.calculate_metric_statistics(combined_df, 'ssim', bitrate_col)
            
            for codec, stats_data in ssim_stats.items():
                # Get consistent color and line style
                color = get_device_color_func(codec) if is_custom_labeled_codec_func(codec) else get_device_color_func(codec)
                codec_type = get_codec_type_func(codec)
                line_style = get_codec_line_style_func(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace
                traces.append({
                    'type': 'scatter',
                    'x': bitrates_kbps.tolist(),
                    'y': stats_data['mean_values'].tolist(),
                    'mode': 'markers+lines',
                    'name': trace_name,
                    'line': {'color': color, 'dash': line_style, 'width': 3},
                    'marker': {'size': 6},
                    'legendgroup': codec,
                    'showlegend': False,
                    'row': 2,
                    'col': 1,
                    'hovertemplate': f"<b>{codec}</b><br>" +
                                   f"Bitrate: %{{x:.1f}} kbps<br>" +
                                   f"SSIM: %{{y:.4f}}<br>" +
                                   f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                   f"Sources: {len(stats_data['sources'])}<br>" +
                                   "<extra></extra>"
                })
                
                # Add confidence interval trace only if we have meaningful confidence intervals
                if not np.allclose(stats_data['upper_bound'], stats_data['lower_bound']):
                    # Convert hex color to rgba with transparency
                    if color.startswith('#'):
                        # Convert hex to rgb
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        fillcolor = f'rgba({r}, {g}, {b}, 0.2)'
                    else:
                        # Fallback to gray if color format is unexpected
                        fillcolor = 'rgba(128, 128, 128, 0.2)'
                    
                    traces.append({
                        'type': 'scatter',
                        'x': bitrates_kbps.tolist() + bitrates_kbps[::-1].tolist(),
                        'y': stats_data['upper_bound'].tolist() + stats_data['lower_bound'][::-1].tolist(),
                        'fill': 'tonexty',
                        'fillcolor': fillcolor,
                        'line': {'color': 'rgba(255,255,255,0)'},
                        'legendgroup': codec,
                        'showlegend': False,
                        'row': 2,
                        'col': 1,
                        'hoverinfo': 'skip'
                    })
        
        return {
            'traces': traces,
            'bitrate_label': bitrate_label
        }
