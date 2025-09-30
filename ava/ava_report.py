#!/usr/bin/env python3

"""
Minimal AVA Reporting System

This module provides basic plotting and reporting capabilities
for video codec test results using Plotly.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import BD-Rate calculation modules
try:
    # Try absolute imports first (when used as script)
    from bd_rate import BDRateCalculator, BDRateResult, compare_codecs_bd_rate, format_bd_rate_result
    from bd_rate_utils import AVABDRateAnalyzer
    from bd_rate_viz import BDRateVisualizer
except ImportError:
    try:
        # Fall back to relative imports (when used as package)
        from .bd_rate import BDRateCalculator, BDRateResult, compare_codecs_bd_rate, format_bd_rate_result
        from .bd_rate_utils import AVABDRateAnalyzer
        from .bd_rate_viz import BDRateVisualizer
    except ImportError as e:
        raise ImportError(f"BD-Rate modules not available: {e}. Please ensure bd_rate.py, bd_rate_utils.py, and bd_rate_viz.py are in the same directory.")


@dataclass
class TestResult:
    """Container for test result data"""
    test_name: str
    device_serial: str
    encoder: str
    success: bool
    model_name: str = ""
    bitrate: int = 0
    source_file: str = ""
    duration: float = 0.0
    quality_metrics: List[Dict[str, Any]] = None
    qp_values: Dict[str, Any] = None
    test_data: Dict[str, Any] = None
    error_message: str = ""

    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = []
        if self.qp_values is None:
            self.qp_values = {}
        if self.test_data is None:
            self.test_data = {}

class ReportGenerator:
    """Generate interactive and static reports from test results"""
    
    def __init__(self, output_dir: str = "reports", debug: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug = debug
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the report generator"""
        logger = logging.getLogger("ava_report")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _prepare_bitrate_data(self, combined_df: pd.DataFrame, bitrate_col: str, bitrate_label: str) -> Dict[str, Any]:
        """Prepare data for a specific bitrate mode"""
        if bitrate_col not in combined_df.columns:
            return {'traces': [], 'bitrate_label': bitrate_label}
        
        traces = []
        
        # VMAF plot data
        if 'vmaf_mean' in combined_df.columns:
            # Calculate VMAF statistics across multiple sources
            vmaf_stats = self._calculate_metric_statistics(combined_df, 'vmaf_mean', bitrate_col)
            
            for codec, stats_data in vmaf_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
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
            psnr_stats = self._calculate_metric_statistics(combined_df, 'psnr', bitrate_col)
            
            for codec, stats_data in psnr_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
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
            ssim_stats = self._calculate_metric_statistics(combined_df, 'ssim', bitrate_col)
            
            for codec, stats_data in ssim_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
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
    
    def create_quality_plots_from_csv(self, test_results: List[TestResult], bitrate_mode: str = "calculated") -> go.Figure:
        """Create quality plots directly from CSV files
        
        Args:
            test_results: List of test results
            bitrate_mode: "calculated" for calculated_bitrate_bps or "target" for bitrate_bps
        """
            
        # Create subplots for different quality metrics
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("VMAF vs Bitrate", "PSNR vs Bitrate", 
                          "SSIM vs Bitrate", "QP Statistics",
                          "Target vs Actual Bitrate", "Bitrate Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Read all quality CSV files directly
        all_data = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'quality_csv' in result.test_data:
                csv_file = result.test_data['quality_csv']
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            # Add device info to the dataframe
                            df['device_serial'] = result.device_serial
                            df['test_name'] = result.test_name
                            
                            # Apply custom labels if this is a custom labeled test
                            df = self._apply_custom_labels_to_dataframe(df, result)
                            
                            all_data.append(df)
                    except pd.errors.EmptyDataError:
                        # Skip empty CSV files (e.g., when quality analysis fails)
                        print(f"Warning: Skipping empty quality CSV file: {csv_file}")
                        continue
        
        if not all_data:
            # Add a message when no quality data is available
            fig.add_annotation(
                text="No quality data available. Quality analysis requires valid encoded video files.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=16, color="red")
            )
            return fig
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Determine which bitrate column to use
        if bitrate_mode == "target":
            bitrate_col = "bitrate_bps"
            bitrate_label = "Target Bitrate (kbps)"
        else:  # calculated
            bitrate_col = "calculated_bitrate_bps"
            bitrate_label = "Calculated Bitrate (kbps)"
        
        # Check if we have the required bitrate column
        if bitrate_col not in combined_df.columns:
            # Add a message when the required bitrate column is not available
            fig.add_annotation(
                text=f"No {bitrate_col} data available. Please ensure quality analysis includes bitrate calculation.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=16, color="red")
            )
            return fig
        
        # VMAF plot - use selected bitrate column for X-axis
        if 'vmaf_mean' in combined_df.columns and bitrate_col in combined_df.columns:
            # Calculate VMAF statistics across multiple sources
            vmaf_stats = self._calculate_metric_statistics(combined_df, 'vmaf_mean', bitrate_col) 
            for codec, stats_data in vmaf_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace (seaborn style - clean line without markers)
                fig.add_trace(
                    go.Scatter(
                        x=bitrates_kbps.tolist(),
                        y=stats_data['mean_values'].tolist(),
                        mode='lines',
                        name=trace_name,
                        line=dict(color=color, width=2.5),
                        legendgroup=codec,
                        showlegend=True,
                        hovertemplate=f"<b>{codec}</b><br>" +
                                     f"Bitrate: %{{x:.1f}} kbps<br>" +
                                     f"VMAF: %{{y:.2f}}<br>" +
                                     f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                     f"Sources: {len(stats_data['sources'])}<br>" +
                                     "<extra></extra>"
                    ),
                    row=1, col=1
                )
                
                # Add confidence interval trace only if we have meaningful confidence intervals
                if not np.allclose(stats_data['upper_bound'], stats_data['lower_bound']):
                    # Convert hex color to rgba with transparency (matching seaborn style)
                    if color.startswith('#'):
                        # Convert hex to rgb
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        fillcolor = f'rgba({r}, {g}, {b}, 0.3)'  # Slightly more opaque like seaborn
                    else:
                        # Fallback to blue like seaborn
                        fillcolor = 'rgba(0, 123, 255, 0.3)'
                    
                    # Create confidence interval fill (seaborn style - properly centered)
                    # Create two traces: upper bound and lower bound with fill between them
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['upper_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['lower_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=fillcolor,
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=1, col=1
                    )
        
        # PSNR plot - use selected bitrate column for X-axis
        if 'psnr' in combined_df.columns and bitrate_col in combined_df.columns:
            # Calculate PSNR statistics across multiple sources
            psnr_stats = self._calculate_metric_statistics(combined_df, 'psnr', bitrate_col)
            
            for codec, stats_data in psnr_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace (seaborn style - clean line without markers)
                fig.add_trace(
                    go.Scatter(
                        x=bitrates_kbps.tolist(),
                        y=stats_data['mean_values'].tolist(),
                        mode='lines',
                        name=trace_name,
                        line=dict(color=color, width=2.5),
                        legendgroup=codec,
                        showlegend=False,
                        hovertemplate=f"<b>{codec}</b><br>" +
                                     f"Bitrate: %{{x:.1f}} kbps<br>" +
                                     f"PSNR: %{{y:.2f}} dB<br>" +
                                     f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                     f"Sources: {len(stats_data['sources'])}<br>" +
                                     "<extra></extra>"
                    ),
                    row=1, col=2
                )
                
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
                    
                    # Create confidence interval fill (seaborn style - properly centered)
                    # Create two traces: upper bound and lower bound with fill between them
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['upper_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['lower_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=fillcolor,
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=1, col=2
                    )
        
        # SSIM plot - use selected bitrate column for X-axis
        if 'ssim' in combined_df.columns and bitrate_col in combined_df.columns:
            # Calculate SSIM statistics across multiple sources
            ssim_stats = self._calculate_metric_statistics(combined_df, 'ssim', bitrate_col)
            
            for codec, stats_data in ssim_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace (seaborn style - clean line without markers)
                fig.add_trace(
                    go.Scatter(
                        x=bitrates_kbps.tolist(),
                        y=stats_data['mean_values'].tolist(),
                        mode='lines',
                        name=trace_name,
                        line=dict(color=color, width=2.5),
                        legendgroup=codec,
                        showlegend=False,
                        hovertemplate=f"<b>{codec}</b><br>" +
                                     f"Bitrate: %{{x:.1f}} kbps<br>" +
                                     f"SSIM: %{{y:.4f}}<br>" +
                                     f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                     f"Sources: {len(stats_data['sources'])}<br>" +
                                     "<extra></extra>"
                    ),
                    row=2, col=1
                )
                
                # Add confidence interval trace only if we have meaningful confidence intervals
                if not np.allclose(stats_data['upper_bound'], stats_data['lower_bound']):
                    # Convert hex color to rgba with transparency (matching seaborn style)
                    if color.startswith('#'):
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        fillcolor = f'rgba({r}, {g}, {b}, 0.3)'  # Slightly more opaque like seaborn
                    else:
                        fillcolor = 'rgba(0, 123, 255, 0.3)'
                    
                    # Create confidence interval fill (seaborn style - properly centered)
                    # Create two traces: upper bound and lower bound with fill between them
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['upper_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['lower_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=fillcolor,
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=2, col=1
                    )
        
        # Target vs Actual Bitrate plot (as line graph)
        if 'bitrate_bps' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device].sort_values('bitrate_bps')
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Get consistent color and line style
                    # Use codec name for color when using custom labels, device name otherwise
                    color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
                    codec_type = self._get_codec_type(codec)
                    line_style = self._get_codec_line_style(codec_type)
                    trace_name = self._get_trace_name(codec, model)
            
                    fig.add_trace(
                        go.Scatter(
                            x=(device_data['bitrate_bps'] / 1000).tolist(),  # Target bitrate
                            y=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Actual bitrate
                            mode='markers+lines',
                            name=trace_name,
                            line=dict(color=color, dash=line_style, width=2),
                            marker=dict(size=6),
                            legendgroup=trace_name,
                            showlegend=False
                        ),
                        row=3, col=1
                    )
        
        # Bitrate Accuracy plot (with averaging and confidence intervals)
        if 'bitrate_bps' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            # Calculate bitrate accuracy for all data
            combined_df['bitrate_accuracy'] = ((combined_df['calculated_bitrate_bps'] - combined_df['bitrate_bps']) / combined_df['bitrate_bps'] * 100)
            
            # Calculate bitrate accuracy statistics across multiple sources
            accuracy_stats = self._calculate_metric_statistics(combined_df, 'bitrate_accuracy', 'bitrate_bps')
            
            for codec, stats_data in accuracy_stats.items():
                # Get consistent color and line style
                color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(codec)
                codec_type = self._get_codec_type(codec)
                line_style = self._get_codec_line_style(codec_type)
                trace_name = f"{codec} (mean)"
                
                # Convert bitrates to kbps for display
                bitrates_kbps = stats_data['bitrates'] / 1000
                
                # Add main line trace (seaborn style - clean line without markers)
                fig.add_trace(
                    go.Scatter(
                        x=bitrates_kbps.tolist(),
                        y=stats_data['mean_values'].tolist(),
                        mode='lines',
                        name=trace_name,
                        line=dict(color=color, width=2.5),
                        legendgroup=codec,
                        showlegend=False,
                        hovertemplate=f"<b>{codec}</b><br>" +
                                     f"Bitrate: %{{x:.1f}} kbps<br>" +
                                     f"Accuracy: %{{y:.2f}}%<br>" +
                                     f"Samples: {stats_data['count'][0] if len(stats_data['count']) > 0 else 'N/A'}<br>" +
                                     f"Sources: {len(stats_data['sources'])}<br>" +
                                     "<extra></extra>"
                    ),
                    row=3, col=2
                )
                
                # Add confidence interval trace only if we have meaningful confidence intervals
                if not np.allclose(stats_data['upper_bound'], stats_data['lower_bound']):
                    # Convert hex color to rgba with transparency (matching seaborn style)
                    if color.startswith('#'):
                        hex_color = color.lstrip('#')
                        r = int(hex_color[0:2], 16)
                        g = int(hex_color[2:4], 16)
                        b = int(hex_color[4:6], 16)
                        fillcolor = f'rgba({r}, {g}, {b}, 0.3)'  # Slightly more opaque like seaborn
                    else:
                        fillcolor = 'rgba(0, 123, 255, 0.3)'
                    
                    # Create confidence interval fill (seaborn style - properly centered)
                    # Create two traces: upper bound and lower bound with fill between them
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['upper_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=3, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=bitrates_kbps.tolist(),
                            y=stats_data['lower_bound'].tolist(),
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=fillcolor,
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=codec
                        ),
                        row=3, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title="Quality Metrics Analysis (Direct from CSV)",
            height=1200,
            showlegend=True,
            autosize=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text=bitrate_label, row=1, col=1)
        fig.update_yaxes(title_text="VMAF", row=1, col=1)
        fig.update_xaxes(title_text=bitrate_label, row=1, col=2)
        fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text=bitrate_label, row=2, col=1)
        fig.update_yaxes(title_text="SSIM", row=2, col=1)
        fig.update_xaxes(title_text="Target Bitrate (kbps)", row=3, col=1)
        fig.update_yaxes(title_text="Actual Bitrate (kbps)", row=3, col=1)
        fig.update_xaxes(title_text="Target Bitrate (kbps)", row=3, col=2)
        fig.update_yaxes(title_text="Bitrate Accuracy (%)", row=3, col=2)
        
        return fig
    
    def _get_codec_type(self, codec_name: str) -> str:
        """Extract codec type from codec name for consistent filtering"""
        codec_lower = codec_name.lower()
        if 'av1' in codec_lower:
            return 'av1'
        elif 'hevc' in codec_lower or 'h265' in codec_lower:
            return 'hevc'
        elif 'avc' in codec_lower or 'h264' in codec_lower:
            return 'avc'
        elif 'vp8' in codec_lower:
            return 'vp8'
        else:
            return 'other'
    
    def _get_device_color(self, device_name: str) -> str:
        """Get consistent color for device across all plots"""
        device_colors = {
            'Pixel 8': '#1f77b4',      # Blue
            'SM-S936U1': '#ff7f0e',    # Orange
            'V2413': '#2ca02c',        # Green
            '10AEBC0MDC001DC': '#d62728', # Red
            'R5CXC2ZH3DR': '#9467bd',   # Purple
            '39251FDJH0093R': '#8c564b',  # Brown
            # Add custom label colors
            'config 1': '#2ca02c',     # Green
            'config 2': '#ff7f0e',     # Orange
            'hevc_v1.0': '#2ca02c',    # Green
            'hevc_v2.0': '#ff7f0e',    # Orange
            'dolby_hevc_v1': '#2ca02c', # Green
            'qualcomm_hevc_v2': '#ff7f0e', # Orange
        }
        
        # If not found in predefined colors, generate a consistent color based on the name
        if device_name not in device_colors:
            # Use hash of the name to generate a consistent color
            import hashlib
            hash_obj = hashlib.md5(device_name.encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            # Generate a color from the hash
            hue = (hash_int % 360) / 360.0
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            return color
        
        return device_colors.get(device_name, '#17becf')  # Default cyan
    
    def _get_codec_line_style(self, codec_type: str) -> str:
        """Get consistent line style for codec type across all plots"""
        codec_line_styles = {
            'av1': 'solid',
            'hevc': 'dash',
            'avc': 'dot',
            'vp8': 'dashdot'
        }
        return codec_line_styles.get(codec_type, 'solid')
    
    def _get_trace_name(self, codec: str, device: str) -> str:
        """Get consistent trace name for filtering"""
        return f"{codec} ({device})"
    
    def _calculate_frame_statistics(self, device_data: pd.DataFrame, metric_column: str) -> tuple:
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
    
    def _is_custom_labeled_codec(self, codec: str) -> bool:
        """Check if a codec name is a custom label (not a standard codec name)"""
        # Custom labels are typically short names like "config 1", "hevc_v1.0", etc.
        # Standard codec names are longer and contain specific patterns
        standard_codec_patterns = [
            'c2.dolby.encoder.hevc',
            'c2.qti.hevc.encoder.hdr',
            'c2.av1.encoder',
            'c2.avc.encoder',
            'c2.vp8.encoder',
            'c2.vp9.encoder'
        ]
        
        # If it matches a standard codec pattern, it's not a custom label
        for pattern in standard_codec_patterns:
            if pattern in codec.lower():
                return False
        
        # If it's a short name (less than 20 characters) and doesn't contain standard patterns, it's likely a custom label
        return len(codec) < 20 and not any(pattern in codec.lower() for pattern in ['encoder', 'codec', 'h264', 'h265', 'av1', 'vp8', 'vp9'])
    

    def _calculate_metric_statistics(self, combined_df: pd.DataFrame, metric_column: str, bitrate_column: str = 'calculated_bitrate_bps') -> dict:
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
            vmaf_stats = self._calculate_metric_statistics(combined_df, 'vmaf_mean', 'calculated_bitrate_bps')
            
            # Calculate PSNR statistics using target bitrate
            psnr_stats = self._calculate_metric_statistics(combined_df, 'psnr', 'bitrate_bps')
            
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
            used_bitrates = set()
            
            for bitrate in unique_bitrates:
                if bitrate in used_bitrates:
                    continue
                
                # Find all bitrates within 5% of this one
                tolerance = 0.05  # 5% tolerance
                similar_bitrates = []
                for other_bitrate in unique_bitrates:
                    if other_bitrate not in used_bitrates:
                        if abs(other_bitrate - bitrate) / bitrate <= tolerance:
                            similar_bitrates.append(other_bitrate)
                            used_bitrates.add(other_bitrate)
                
                # Use the mean of similar bitrates as the representative bitrate
                representative_bitrate = np.mean(similar_bitrates)
                bitrate_groups.append((representative_bitrate, similar_bitrates))
            
            # Calculate statistics for each bitrate group, averaging across different reference files
            bitrate_values = []
            mean_values = []
            upper_bounds = []
            lower_bounds = []
            counts = []
            sources = []
            
            for representative_bitrate, similar_bitrates in bitrate_groups:
                # Get all data points at these similar bitrates (across all reference files and devices)
                bitrate_mask = codec_data[bitrate_column].isin(similar_bitrates)
                bitrate_data = codec_data[bitrate_mask]
                
                if bitrate_data.empty:
                    continue
                
                # Calculate statistics across all sources (devices) and reference files at this bitrate
                metric_values = bitrate_data[metric_column].values
                mean_val = np.mean(metric_values)
                count = len(metric_values)
                
                # Calculate confidence intervals
                # With 2 or more data points, we can show some indication of variability
                if count < 2:
                    # No confidence interval for single data point
                    upper_bound = mean_val
                    lower_bound = mean_val
                else:
                    # Calculate proper confidence intervals
                    std_val = np.std(metric_values, ddof=1)  # Sample standard deviation
                    
                    # Handle case where all values are the same (std = 0)
                    if std_val == 0:
                        std_val = mean_val * 0.01  # Use 1% of mean as small std
                    
                    # Calculate confidence interval using seaborn's bootstrap approach
                    # For small samples, use a simplified bootstrap-like method
                    if count >= 3:
                        # Use bootstrap-style confidence interval
                        # Resample the data multiple times and calculate mean
                        n_bootstrap = min(1000, count * 100)  # Bootstrap samples
                        bootstrap_means = []
                        for _ in range(n_bootstrap):
                            # Sample with replacement
                            bootstrap_sample = np.random.choice(metric_values, size=count, replace=True)
                            bootstrap_means.append(np.mean(bootstrap_sample))
                        
                        # Calculate 95% confidence interval from bootstrap distribution
                        lower_bound = np.percentile(bootstrap_means, 2.5)
                        upper_bound = np.percentile(bootstrap_means, 97.5)
                    else:
                        # For very small samples (2 data points), use a conservative approach
                        # Use the range of the data as a rough confidence interval
                        data_range = np.max(metric_values) - np.min(metric_values)
                        half_range = data_range * 0.25  # Conservative 25% of range
                        upper_bound = mean_val + half_range
                        lower_bound = mean_val - half_range
                    
                    # Ensure bounds are reasonable for the metric while maintaining symmetry
                    if metric_column in ['psnr', 'ssim', 'vmaf_mean']:
                        if metric_column == 'vmaf_mean':
                            # Keep confidence interval symmetric around mean
                            half_range = min(mean_val - 0, 100 - mean_val)  # Distance to nearest boundary
                            actual_half_range = min(mean_val - lower_bound, upper_bound - mean_val)  # Actual CI half-range
                            symmetric_range = min(half_range, actual_half_range)  # Don't exceed boundaries
                            upper_bound = mean_val + symmetric_range
                            lower_bound = mean_val - symmetric_range
                        elif metric_column == 'psnr':
                            # Keep confidence interval symmetric around mean
                            half_range = min(mean_val - 0, 100 - mean_val)  # Distance to nearest boundary (assuming max PSNR ~100)
                            actual_half_range = min(mean_val - lower_bound, upper_bound - mean_val)
                            symmetric_range = min(half_range, actual_half_range)
                            upper_bound = mean_val + symmetric_range
                            lower_bound = mean_val - symmetric_range
                        elif metric_column == 'ssim':
                            # Keep confidence interval symmetric around mean
                            half_range = min(mean_val - 0, 1 - mean_val)  # Distance to nearest boundary
                            actual_half_range = min(mean_val - lower_bound, upper_bound - mean_val)
                            symmetric_range = min(half_range, actual_half_range)
                            upper_bound = mean_val + symmetric_range
                            lower_bound = mean_val - symmetric_range
                
                
                bitrate_values.append(representative_bitrate)
                mean_values.append(mean_val)
                upper_bounds.append(upper_bound)
                lower_bounds.append(lower_bound)
                counts.append(count)
                
                # Collect source identifiers (devices and reference files)
                source_ids = bitrate_data['device_serial'].unique().tolist()
                reference_files = bitrate_data['reference_file'].unique().tolist()
                sources.extend(source_ids)
                sources.extend(reference_files)  # Also include reference files as sources
            
            if len(bitrate_values) > 0:
                results[codec] = {
                    'bitrates': np.array(bitrate_values),
                    'mean_values': np.array(mean_values),
                    'upper_bound': np.array(upper_bounds),
                    'lower_bound': np.array(lower_bounds),
                    'count': np.array(counts),
                    'sources': list(set(sources))  # Unique sources (devices and reference files)
                }
        
        return results


    def create_performance_plots(self, test_results: List[TestResult]) -> go.Figure:
        """Create performance plots from encoder statistics CSV files"""
        
        # Create subplots for performance metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Encoding Latency (Frame-based)", "Processing Framerate (Frame-based)", 
                          "Pipeline Depth Over Time", "Bitrate (5-frame Moving Average)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Load all encoding data CSV files
        all_data = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'stats_csv' in result.test_data:
                csv_files = result.test_data['stats_csv']
                if isinstance(csv_files, list):
                    for csv_file in csv_files:
                        if csv_file.endswith('_encoding_data.csv') and os.path.exists(csv_file):
                            try:
                                df = pd.read_csv(csv_file)
                                if not df.empty:
                                    # Add device info to the dataframe
                                    df['device_serial'] = result.device_serial
                                    df['test_name'] = result.test_name
                                    all_data.append(df)
                            except pd.errors.EmptyDataError:
                                # Skip empty CSV files
                                print(f"Warning: Skipping empty performance CSV file: {csv_file}")
                                continue
        
        if not all_data:
            # If no encoder stats, create placeholder plots
            fig.add_annotation(
                text="No encoder statistics data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Plot 1: Aggregated Encoding Latency
        if 'proctime' in combined_df.columns and 'frame' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Convert nanoseconds to milliseconds first, then calculate frame statistics
                    device_data_copy = device_data.copy()
                    device_data_copy['proctime_ms'] = device_data_copy['proctime'] / 1_000_000
                    
                    # Calculate frame-based statistics
                    time_sec, mean_latency, upper_bound, lower_bound, count = self._calculate_frame_statistics(device_data_copy, 'proctime_ms')
                    
                    # Create consistent trace naming for filtering
                    trace_name = self._get_trace_name(codec, model)
                    codec_type = self._get_codec_type(codec)
                    
                    # Get consistent color and line style
                    # Use codec name for color when using custom labels, device name otherwise
                    color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
                    line_style = self._get_codec_line_style(codec_type)
                    
                    # Add main line
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=mean_latency.tolist(),
                            mode='lines+markers',
                            name=trace_name,
                            line=dict(width=2, color=color, dash=line_style),
                            marker=dict(color=color, size=4),
                            legendgroup=trace_name,
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
                    # Add confidence interval (upper bound)
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=upper_bound.tolist(),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=trace_name,
                            visible='legendonly'
                    ),
                    row=1, col=1
                )
        
                    # Add confidence interval (lower bound with fill)
                fig.add_trace(
                    go.Scatter(
                            x=time_sec.tolist(),
                            y=lower_bound.tolist(),
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=trace_name,
                            visible='legendonly'
                        ),
                        row=1, col=1
                    )
        
        # Plot 2: Processing Framerate
        if 'proc_fps' in combined_df.columns and 'frame' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Calculate frame-based statistics
                    time_sec, mean_fps, upper_bound, lower_bound, count = self._calculate_frame_statistics(device_data, 'proc_fps')
                    
                    # Create consistent trace naming for filtering
                    trace_name = self._get_trace_name(codec, model)
                    codec_type = self._get_codec_type(codec)
                    
                    # Get consistent color and line style
                    # Use codec name for color when using custom labels, device name otherwise
                    color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
                    line_style = self._get_codec_line_style(codec_type)
                    
                    # Add main line
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=mean_fps.tolist(),
                            mode='lines+markers',
                            name=trace_name,
                            line=dict(width=2, color=color, dash=line_style),
                            marker=dict(color=color, size=4),
                            legendgroup=trace_name,
                            showlegend=True
                        ),
                        row=1, col=2
                    )
                    
                    # Add confidence interval (upper bound)
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=upper_bound.tolist(),
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=trace_name,
                            visible='legendonly'
                        ),
                        row=1, col=2
                    )
                    
                    # Add confidence interval (lower bound with fill)
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=lower_bound.tolist(),
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=trace_name,
                            visible='legendonly'
                        ),
                        row=1, col=2
                    )
        
        # Plot 3: Pipeline Depth Over Time
        if 'inflight' in combined_df.columns and 'rel_pts' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    inflight = device_data['inflight'].values
                    time_sec = (device_data['rel_pts'] / 1000).values
                    
                    # Sort by time for proper aggregation
                    sort_idx = np.argsort(time_sec)
                    time_sec = time_sec[sort_idx]
                    inflight = inflight[sort_idx]
                    
                    # Create consistent trace naming for filtering
                    trace_name = self._get_trace_name(codec, model)
                    codec_type = self._get_codec_type(codec)
                    
                    # Get consistent color and line style
                    # Use codec name for color when using custom labels, device name otherwise
                    color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
                    line_style = self._get_codec_line_style(codec_type)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=inflight.tolist(),
                            mode='lines',
                            name=trace_name,
                            line=dict(width=2, color=color, dash=line_style),
                            legendgroup=trace_name,
                            showlegend=False
                        ),
                        row=2, col=1
                    )
        
        # Plot 4: Bitrate Variability (separate by target bitrate)
        if 'bitrate_per_frame_bps' in combined_df.columns and 'frame' in combined_df.columns:
            # Group by target bitrate first
            for target_bitrate in combined_df['bitrate'].unique():
                target_data = combined_df[combined_df['bitrate'] == target_bitrate]
                target_bitrate_kbps = target_bitrate / 1000
                
                print(f"DEBUG: Processing target bitrate: {target_bitrate_kbps:.0f} kbps")
                
                for codec in target_data['codec'].unique():
                    codec_data = target_data[target_data['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device]
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                        
                        # Get fps
                        fps = device_data['fps'].iloc[0] if 'fps' in device_data.columns else 30
                        
                        # Group by frame and calculate statistics across all videos
                        frame_stats = device_data.groupby('frame').agg({
                            'bitrate_per_frame_bps': ['mean', 'std', 'count']
                        }).reset_index()
                        
                        # Flatten column names
                        frame_stats.columns = ['frame', 'mean_bitrate', 'std_bitrate', 'count']
                        
                        # Apply 5-frame moving average to the mean bitrate
                        bitrate_series = pd.Series(frame_stats['mean_bitrate'])
                        moving_avg = bitrate_series.rolling(window=5, center=True, min_periods=1).mean()
                        
                        # Apply 5-frame moving average to the std as well
                        std_series = pd.Series(frame_stats['std_bitrate'])
                        moving_std = std_series.rolling(window=5, center=True, min_periods=1).mean()
                        
                        # Convert to time and kbps
                        time_sec = (frame_stats['frame'] / fps).values
                        bitrate_kbps = (moving_avg / 1000).values
                        std_kbps = (moving_std / 1000).values
                        
                        # Calculate confidence interval
                        confidence_factor = 1.96  # 95% CI
                        upper_bound = bitrate_kbps + confidence_factor * std_kbps
                        lower_bound = bitrate_kbps - confidence_factor * std_kbps
                        
                        # Line thickness based on target bitrate (higher bitrate = thicker line)
                        line_width = max(2, min(12, 2 + target_bitrate_kbps / 500))  # 2-12 pixel width based on target bitrate
                        
                        print(f"DEBUG: {codec} on {model}: {len(device_data)} videos, line width: {line_width:.1f}")
                        print(f"DEBUG: Bitrate range: {bitrate_kbps.min():.2f} to {bitrate_kbps.max():.2f} kbps")
                        
                        # Create simple trace name (device/codec only, no bitrate suffix)
                        trace_name = f"{codec} ({model})"
                        
                        # Get consistent color (no line style variation)
                        # Use codec name for color when using custom labels, device name otherwise
                        color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
            
                        # Only show legend for the first target bitrate of each device/codec combination
                        show_legend = bool(target_bitrate == combined_df[combined_df['codec'] == codec]['bitrate'].min())
            
                        # Plot the averaged 5-frame moving average bitrate data
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec.tolist(),
                                y=bitrate_kbps.tolist(),
                                mode='lines',
                                name=trace_name,
                                line=dict(color=color, width=line_width),
                                legendgroup=trace_name,
                                showlegend=show_legend,
                                connectgaps=False,
                                hovertemplate=f'<b>{codec} ({model})</b><br>' +
                                            f'Target: {target_bitrate_kbps:.0f} kbps<br>' +
                                            'Time: %{x:.3f}s<br>' +
                                            'Bitrate: %{y:.0f} kbps<br>' +
                                            '<extra></extra>'
                            ),
                            row=2, col=2
                        )
                        
                        # Add confidence interval (upper bound)
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec.tolist(),
                                y=upper_bound.tolist(),
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip',
                                legendgroup=trace_name,
                                visible='legendonly'
                            ),
                            row=2, col=2
                        )
                        
                        # Add confidence interval (lower bound with fill)
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec.tolist(),
                                y=lower_bound.tolist(),
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                                showlegend=False,
                                hoverinfo='skip',
                                legendgroup=trace_name,
                                visible='legendonly'
                            ),
                            row=2, col=2
                        )
        
        # Update layout
        fig.update_layout(
            title="Performance Analysis - Encoder Statistics",
            height=800,
            showlegend=True,
            autosize=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (sec)", row=1, col=1)
        fig.update_yaxes(title_text="Latency (msec)", row=1, col=1)
        fig.update_xaxes(title_text="Time (sec)", row=1, col=2)
        fig.update_yaxes(title_text="Average processing fps", row=1, col=2)
        fig.update_xaxes(title_text="Time (sec)", row=2, col=1)
        fig.update_yaxes(title_text="Inflight", row=2, col=1)
        fig.update_xaxes(title_text="Time (sec)", row=2, col=2)
        fig.update_yaxes(title_text="Bitrate per Frame (kbps)", row=2, col=2)
        
        return fig
    
    def create_comparison_plots(self, test_results: List[TestResult], reference_pair: str = None, bitrate_mode: str = "calculated") -> go.Figure:
        """Create comparison plots with delta analysis"""
        
        # Create single plot for VMAF delta comparison only
        fig = go.Figure()
        
        # Read all quality CSV files directly
        all_data = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'quality_csv' in result.test_data:
                csv_file = result.test_data['quality_csv']
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            # Add device info to the dataframe
                            df['device_serial'] = result.device_serial
                            df['test_name'] = result.test_name
                            
                            # Apply custom labels if this is a custom labeled test
                            df = self._apply_custom_labels_to_dataframe(df, result)
                            
                            all_data.append(df)
                    except pd.errors.EmptyDataError:
                        # Skip empty CSV files (e.g., when quality analysis fails)
                        print(f"Warning: Skipping empty quality CSV file: {csv_file}")
                        continue
        
        if not all_data:
            return fig
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Determine which bitrate column to use
        if bitrate_mode == "target":
            bitrate_col = "bitrate_bps"
        else:  # calculated
            bitrate_col = "calculated_bitrate_bps"
        
        # Check if we have quality data
        if 'vmaf_mean' not in combined_df.columns or bitrate_col not in combined_df.columns:
            # Add a message when no quality data is available
            fig.add_annotation(
                text="No quality data available. Quality analysis requires valid encoded video files.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=16, color="red")
            )
            return fig
        
        # VMAF Delta Comparison
        if 'vmaf_mean' in combined_df.columns and bitrate_col in combined_df.columns:
            # Choose reference based on selection or first available
            reference_data = None
            if reference_pair:
                # Parse the reference pair (format: "codec (device)")
                import re
                match = re.match(r'([^(]+)\s*\(([^)]+)\)', reference_pair)
                if match:
                    ref_codec = match.group(1).strip()
                    ref_device = match.group(2).strip()
                    
                    # Find reference data for selected codec-device pair
                    codec_data = combined_df[combined_df['codec'] == ref_codec]
                    device_data = codec_data[codec_data['device_serial'] == ref_device].sort_values(bitrate_col)
                    if len(device_data) >= 2:
                        reference_data = device_data
            
            # Fallback to first available if no reference selected or found
            if reference_data is None:
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device].sort_values(bitrate_col)
                        if len(device_data) >= 2:
                            reference_data = device_data
                            break
                    if reference_data is not None:
                        break
            
            if reference_data is not None:
                # Interpolate reference curve
                ref_bitrates = reference_data[bitrate_col].values
                ref_vmaf = reference_data['vmaf_mean'].values
                
                # Add the reference trace itself (shows as 0 delta)
                ref_codec = reference_data['codec'].iloc[0]
                ref_device = reference_data['device_serial'].iloc[0]
                ref_model = reference_data['model'].iloc[0] if 'model' in reference_data.columns else ref_device
                
                # Get consistent color and line style for reference
                color = self._get_device_color(ref_codec) if self._is_custom_labeled_codec(ref_codec) else self._get_device_color(ref_model)
                codec_type = self._get_codec_type(ref_codec)
                line_style = self._get_codec_line_style(codec_type)
                trace_name = self._get_trace_name(ref_codec, ref_model)
                
                # Add reference trace (0 delta)
                fig.add_trace(
                    go.Scatter(
                        x=(ref_bitrates / 1000).tolist(),
                        y=np.zeros_like(ref_bitrates).tolist(),  # 0 delta for reference
                        mode='markers+lines',
                        name=trace_name,
                        line=dict(color=color, dash=line_style, width=2),
                        marker=dict(size=6),
                        legendgroup=trace_name,
                        showlegend=True
                    )
                )
                
                # Create interpolation function
                ref_interp = np.interp
                
                # Compare all other combinations (skip the reference)
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device].sort_values(bitrate_col)
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                        
                        # Skip the reference combination (already added above)
                        if (codec == ref_codec and device == ref_device):
                            continue
                        
                        if len(device_data) >= 2:
                            # Calculate delta
                            bitrates = device_data[bitrate_col].values
                            vmaf_values = device_data['vmaf_mean'].values
                            
                            # Interpolate reference VMAF at same bitrates
                            ref_vmaf_interp = np.interp(bitrates, ref_bitrates, ref_vmaf)
                            delta_vmaf = vmaf_values - ref_vmaf_interp
        
                            # Get consistent color and line style
                            # Use codec name for color when using custom labels, device name otherwise
                            color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
                            codec_type = self._get_codec_type(codec)
                            line_style = self._get_codec_line_style(codec_type)
                            trace_name = self._get_trace_name(codec, model)
        
                            fig.add_trace(
                                go.Scatter(
                                    x=(bitrates / 1000).tolist(),
                                    y=delta_vmaf.tolist(),
                                    mode='markers+lines',
                                    name=trace_name,
                                    line=dict(color=color, dash=line_style, width=2),
                                    marker=dict(size=6),
                                    legendgroup=trace_name,
                                    showlegend=True
                                )
                            )
        
        
        # Update layout
        fig.update_layout(
            title="VMAF Delta Comparison",
            height=600,
            showlegend=True,
            autosize=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Actual Bitrate (kbps)")
        fig.update_yaxes(title_text="VMAF Delta")
        
        return fig
    
    def create_siti_plots(self, test_results: List[TestResult]) -> go.Figure:
        """Create SI/TI complexity analysis plots with four different visualizations"""
        
        # Read all quality CSV files directly
        all_data = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'quality_csv' in result.test_data:
                csv_file = result.test_data['quality_csv']
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            # Add device info to the dataframe
                            df['device_serial'] = result.device_serial
                            df['test_name'] = result.test_name
                            
                            # Apply custom labels if this is a custom labeled test
                            df = self._apply_custom_labels_to_dataframe(df, result)
                            
                            all_data.append(df)
                    except pd.errors.EmptyDataError:
                        # Skip empty CSV files (e.g., when quality analysis fails)
                        print(f"Warning: Skipping empty quality CSV file: {csv_file}")
                        continue
        
        if not all_data:
            # Create empty plot if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No SI/TI data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="SI/TI Complexity Analysis",
                height=600,
                showlegend=True
            )
            return fig
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create subplots with 2x2 layout
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("SI vs TI Distribution", "SI Distribution", "TI Distribution", "Source Count by Complexity Level"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # SI/TI Complexity Analysis
        if 'si_avg' in combined_df.columns and 'ti_avg' in combined_df.columns:
            # Filter out invalid values
            valid_data = combined_df[(combined_df['si_avg'] != -1) & (combined_df['ti_avg'] != -1)]
            
            if not valid_data.empty:
                # Collect data for analysis
                si_values = []
                ti_values = []
                source_info = []
                
                for codec in valid_data['codec'].unique():
                    codec_data = valid_data[valid_data['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device]
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                        
                        # Get consistent color and line style
                        # Use codec name for color when using custom labels, device name otherwise
                        color = self._get_device_color(codec) if self._is_custom_labeled_codec(codec) else self._get_device_color(model)
                        codec_type = self._get_codec_type(codec)
                        trace_name = self._get_trace_name(codec, model)
                        
                        # Collect data points
                        for _, row in device_data.iterrows():
                            si_values.append(row['si_avg'])
                            ti_values.append(row['ti_avg'])
                            source_info.append({
                                'codec': codec,
                                'model': model,
                                'trace_name': trace_name,
                                'color': color
                            })
                
                # 1. SI vs TI Distribution (top-left)
                if si_values and ti_values:
                    # Group by trace for coloring
                    trace_data = {}
                    for i, (si, ti, info) in enumerate(zip(si_values, ti_values, source_info)):
                        trace_name = info['trace_name']
                        if trace_name not in trace_data:
                            trace_data[trace_name] = {'si': [], 'ti': [], 'color': info['color']}
                        trace_data[trace_name]['si'].append(si)
                        trace_data[trace_name]['ti'].append(ti)
                    
                    for trace_name, data in trace_data.items():
                        fig.add_trace(
                            go.Scatter(
                                x=data['ti'],
                                y=data['si'],
                                mode='markers',
                                name=trace_name,
                                marker=dict(size=8, color=data['color']),
                                hovertemplate=f'<b>{trace_name}</b><br>' +
                                            'TI: %{{x}}<br>' +
                                            'SI: %{{y}}<br>' +
                                            '<extra></extra>',
                                legendgroup=trace_name,
                                showlegend=True
                            ),
                            row=1, col=1
                        )
                    
                    # Add grid lines to show complexity levels (5x5 grid)
                    # SI levels: 0, 20, 40, 60, 80, 100, 120
                    for si_level in [0, 20, 40, 60, 80, 100, 120]:
                        fig.add_hline(
                            y=si_level,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.3,
                            row=1, col=1
                        )
                    
                    # TI levels: 0, 20, 40, 60, 80, 100
                    for ti_level in [0, 20, 40, 60, 80, 100]:
                        fig.add_vline(
                            x=ti_level,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.3,
                            row=1, col=1
                        )
                    
                    # 2. SI Distribution (top-right)
                    fig.add_trace(
                        go.Histogram(
                            x=si_values,
                            name='SI Distribution',
                            marker_color='lightblue',
                            opacity=0.7
                        ),
                        row=1, col=2
                    )
                    
                    # 3. TI Distribution (bottom-left)
                    fig.add_trace(
                        go.Histogram(
                            x=ti_values,
                            name='TI Distribution',
                            marker_color='lightcoral',
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
                    
                    # 4. Source Count by Complexity Level (bottom-right)
                    # Categorize sources by complexity levels
                    complexity_counts = {}
                    for si, ti in zip(si_values, ti_values):
                        # Categorize SI
                        if si < 20:
                            si_cat = 'vl'
                        elif si < 40:
                            si_cat = 'l'
                        elif si < 60:
                            si_cat = 'm'
                        elif si < 80:
                            si_cat = 'h'
                        else:
                            si_cat = 'vh'
                        
                        # Categorize TI
                        if ti < 20:
                            ti_cat = 'vl'
                        elif ti < 40:
                            ti_cat = 'l'
                        elif ti < 60:
                            ti_cat = 'm'
                        elif ti < 80:
                            ti_cat = 'h'
                        else:
                            ti_cat = 'vh'
                        
                        key = f"{si_cat}_{ti_cat}"
                        complexity_counts[key] = complexity_counts.get(key, 0) + 1
                    
                    # Create heatmap data
                    si_categories = ['vl', 'l', 'm', 'h', 'vh']
                    ti_categories = ['vl', 'l', 'm', 'h', 'vh']
                    
                    heatmap_data = []
                    for si_cat in si_categories:
                        row = []
                        for ti_cat in ti_categories:
                            key = f"{si_cat}_{ti_cat}"
                            row.append(complexity_counts.get(key, 0))
                        heatmap_data.append(row)
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=heatmap_data,
                            x=ti_categories,
                            y=si_categories,
                            colorscale='Viridis',
                            showscale=True,
                            name='Source Count'
                        ),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title="SI/TI Complexity Analysis - Test Results Distribution",
            height=800,
            showlegend=True,
            autosize=True,
            dragmode='zoom',
            hovermode='closest'
        )
        
        # Update axes labels and ranges
        fig.update_xaxes(title_text="TI (Temporal Information)", range=[0, 120], row=1, col=1)
        fig.update_yaxes(title_text="SI (Spatial Information)", range=[0, 120], row=1, col=1)
        fig.update_xaxes(title_text="SI Values", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="TI Values", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Temporal Complexity", row=2, col=2)
        fig.update_yaxes(title_text="Spatial Complexity", row=2, col=2)
        
        return fig
    
    def _calculate_si_ti(self, video_file: str) -> Tuple[float, float]:
        """Calculate SI/TI values for a video file using FFmpeg"""
        import tempfile
        import subprocess
        
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
            print(f"Error calculating SI/TI for {video_file}: {e}")
            return -1.0, -1.0
    
    def generate_interactive_report(self, test_results: List[TestResult], 
                                  stats_files: List[str] = None, bitrate_mode: str = "calculated") -> str:
        """Generate an interactive HTML report
        
        Args:
            test_results: List of test results
            stats_files: Optional list of stats files
            bitrate_mode: "calculated" for calculated_bitrate_bps or "target" for bitrate_bps
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ava_report_{timestamp}.html"
        
        # Create plots for both bitrate modes
        quality_fig_calculated = self.create_quality_plots_from_csv(test_results, "calculated")
        quality_fig_target = self.create_quality_plots_from_csv(test_results, "target")
        
        # Use the requested mode as default
        quality_fig = quality_fig_calculated if bitrate_mode == "calculated" else quality_fig_target
        performance_fig = self.create_performance_plots(test_results)
        comparison_fig = self.create_comparison_plots(test_results, bitrate_mode=bitrate_mode)
        siti_fig = self.create_siti_plots(test_results)
        
        # For dynamic BD-Rate, we'll generate plots in JavaScript
        # No need to pre-calculate all combinations
        bd_visualizations = {}
        self.logger.info("BD-Rate visualizations will be generated dynamically in JavaScript")
        
        # Generate HTML content with tabs
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AVA Quality Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                .tab {{
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                }}
                .tab button:hover {{
                    background-color: #ddd;
                }}
                .tab button.active {{
                    background-color: #ccc;
                }}
                .tabcontent {{
                    display: none;
                    padding: 6px 12px;
                    border: 1px solid #ccc;
                    border-top: none;
                }}
                .tabcontent.active {{
                    display: block;
                }}
            </style>
        </head>
        <body>
            <h1>AVA Quality Metrics Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <!-- Bitrate Mode Toggle -->
            <div style="margin: 20px 0; padding: 10px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;">
                <label for="bitrateMode" style="font-weight: bold; margin-right: 10px;">Bitrate Mode:</label>
                <select id="bitrateMode" onchange="toggleBitrateMode()" style="padding: 5px; font-size: 14px;">
                    <option value="calculated">Calculated Bitrate</option>
                    <option value="target">Target Bitrate</option>
                </select>
                <span id="bitrateModeDescription" style="margin-left: 10px; color: #666;">
                    Shows actual bitrate calculated from encoded video files
                </span>
            </div>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'Quality')">Quality Metrics</button>
                <button class="tablinks" onclick="openTab(event, 'Performance')">Performance</button>
                <button class="tablinks" onclick="openTab(event, 'Comparison')">Comparison</button>
                <button class="tablinks" onclick="openTab(event, 'SI-TI')">SI/TI Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'BD-Rate')">BD-Rate Analysis</button>
            </div>
            
            <div id="Quality" class="tabcontent active">
                <div style="margin-bottom: 20px;">
                    <h3>Filters</h3>
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        <div>
                            <h4>Codec Types:</h4>
                            <label><input type="checkbox" id="filter-av1" checked onchange="filterTraces()"> AV1</label><br>
                            <label><input type="checkbox" id="filter-hevc" checked onchange="filterTraces()"> HEVC/H.265</label><br>
                            <label><input type="checkbox" id="filter-avc" checked onchange="filterTraces()"> AVC/H.264</label><br>
                            <label><input type="checkbox" id="filter-vp8" checked onchange="filterTraces()"> VP8</label>
                        </div>
                        <div>
                            <h4>Devices:</h4>
                            <div id="device-filters"></div>
                        </div>
                        <div>
                            <h4>Data Information:</h4>
                            <div id="dataInfoNote" style="background: #e7f3ff; padding: 10px; border-radius: 4px; border-left: 4px solid #2196F3;">
                                <strong>Note:</strong> VMAF values are plotted against actual achieved bitrates, not target bitrates. 
                                This automatically accounts for bitrate accuracy differences between devices.
                            </div>
                        </div>
                        <div>
                            <h4>Actions:</h4>
                            <button onclick="showAllTraces()">Show All</button>
                            <button onclick="hideAllTraces()">Hide All</button>
                        </div>
                    </div>
                </div>
                <div id="quality-plots"></div>
            </div>
            
            <div id="Performance" class="tabcontent">
                <div style="margin-bottom: 20px;">
                    <h3>Filters</h3>
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        <div>
                            <h4>Codec Types:</h4>
                            <label><input type="checkbox" id="perf-filter-av1" checked onchange="filterPerformanceTraces()"> AV1</label><br>
                            <label><input type="checkbox" id="perf-filter-hevc" checked onchange="filterPerformanceTraces()"> HEVC/H.265</label><br>
                            <label><input type="checkbox" id="perf-filter-avc" checked onchange="filterPerformanceTraces()"> AVC/H.264</label><br>
                            <label><input type="checkbox" id="perf-filter-vp8" checked onchange="filterPerformanceTraces()"> VP8</label>
                        </div>
                        <div>
                            <h4>Devices:</h4>
                            <div id="perf-device-filters"></div>
                        </div>
                        <div>
                            <h4>Actions:</h4>
                            <button onclick="showAllPerformanceTraces()">Show All</button>
                            <button onclick="hideAllPerformanceTraces()">Hide All</button>
                        </div>
                    </div>
                </div>
                <div id="performance-plots"></div>
            </div>
            
            <div id="Comparison" class="tabcontent">
                <div style="margin-bottom: 20px;">
                    <h3>Filters</h3>
                    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                        <div>
                            <h4>Codec Types:</h4>
                            <label><input type="checkbox" id="comp-filter-av1" checked onchange="filterComparisonTraces()"> AV1</label><br>
                            <label><input type="checkbox" id="comp-filter-hevc" checked onchange="filterComparisonTraces()"> HEVC/H.265</label><br>
                            <label><input type="checkbox" id="comp-filter-avc" checked onchange="filterComparisonTraces()"> AVC/H.264</label><br>
                            <label><input type="checkbox" id="comp-filter-vp8" checked onchange="filterComparisonTraces()"> VP8</label>
                        </div>
                        <div>
                            <h4>Devices:</h4>
                            <div id="comp-device-filters"></div>
                        </div>
                        <div>
                            <h4>Reference Selection:</h4>
                            <select id="reference-selector" onchange="updateReference()">
                                <option value="">Select Reference...</option>
                            </select>
                        </div>
                        <div>
                            <h4>Data Information:</h4>
                            <div style="background: #e7f3ff; padding: 10px; border-radius: 4px; border-left: 4px solid #2196F3;">
                                <strong>Note:</strong> VMAF deltas are calculated from actual achieved bitrates, not target bitrates. 
                                This automatically accounts for bitrate accuracy differences between devices.
                            </div>
                        </div>
                        <div>
                            <h4>Actions:</h4>
                            <button onclick="showAllComparisonTraces()">Show All</button>
                            <button onclick="hideAllComparisonTraces()">Hide All</button>
                        </div>
                    </div>
                </div>
                <div id="comparison-plots"></div>
            </div>
            
            <div id="SI-TI" class="tabcontent">
                <div id="siti-plots"></div>
            </div>
            
            <div id="BD-Rate" class="tabcontent">
                <div id="bd-rate-content">
                    <p>BD-Rate analysis will be loaded here...</p>
                </div>
            </div>
            
            <script>
                // Global variables for plot data
                var qualityDataCalculated = {quality_fig_calculated.to_json()};
                var qualityDataTarget = {quality_fig_target.to_json()};
                var qualityData = qualityDataCalculated; // Default to calculated
                
                // Get bitrate mode from URL parameter or use default
                var urlParams = new URLSearchParams(window.location.search);
                var currentBitrateMode = urlParams.get('bitrate_mode') || '{bitrate_mode}';
                
                // Set the dropdown to the correct value based on URL parameter
                function setBitrateMode() {{
                    console.log('Setting bitrate mode to:', currentBitrateMode);
                    var bitrateModeSelect = document.getElementById('bitrateMode');
                    if (bitrateModeSelect) {{
                        console.log('Found bitrate select element');
                        bitrateModeSelect.value = currentBitrateMode;
                        console.log('Set select value to:', bitrateModeSelect.value);
                        
                        // Set the correct dataset based on mode
                        if (currentBitrateMode === 'target') {{
                            qualityData = qualityDataTarget;
                        }} else {{
                            qualityData = qualityDataCalculated;
                        }}
                        
                        // Update the description text
                        var description = document.getElementById('bitrateModeDescription');
                        if (description) {{
                            if (currentBitrateMode === 'calculated') {{
                                description.textContent = 'Shows actual bitrate calculated from encoded video files';
                            }} else {{
                                description.textContent = 'Shows target bitrate specified for encoding';
                            }}
                            console.log('Updated description text');
                        }}
                        
                        // Update the data info note
                        var note = document.getElementById('dataInfoNote');
                        if (note) {{
                            if (currentBitrateMode === 'calculated') {{
                                note.innerHTML = '<strong>Note:</strong> VMAF values are plotted against actual achieved bitrates, not target bitrates. This automatically accounts for bitrate accuracy differences between devices.';
                            }} else {{
                                note.innerHTML = '<strong>Note:</strong> VMAF values are plotted against target bitrates specified for encoding. This shows the intended bitrate vs actual quality relationship.';
                            }}
                            console.log('Updated data info note');
                        }}
                    }} else {{
                        console.log('Bitrate select element not found');
                    }}
                }}
                
                // Try to set immediately, then also on DOM ready
                setBitrateMode();
                document.addEventListener('DOMContentLoaded', setBitrateMode);
                var performanceData = {performance_fig.to_json()};
                var comparisonData = {comparison_fig.to_json()};
                var sitiData = {siti_fig.to_json()};
                
                // Current bitrate mode
                var currentBitrateMode = '{bitrate_mode}';
                
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].classList.remove("active");
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].classList.remove("active");
                    }}
                    document.getElementById(tabName).classList.add("active");
                    evt.currentTarget.classList.add("active");
                    
                    // Initialize BD-Rate when that tab is clicked
                    if (tabName === 'BD-Rate' && typeof initializeBDRate === 'function') {{
                        console.log('🔄 BD-Rate tab clicked, initializing...');
                        initializeBDRate();
                    }}
                    
                    // Initialize SI/TI when that tab is clicked
                    if (tabName === 'SI-TI' && typeof initializeSITI === 'function') {{
                        console.log('🔄 SI/TI tab clicked, initializing...');
                        initializeSITI();
                    }}
                }}
                
                function toggleBitrateMode() {{
                    var mode = document.getElementById('bitrateMode').value;
                    var description = document.getElementById('bitrateModeDescription');
                    var note = document.getElementById('dataInfoNote');
                    
                    console.log('Switching to bitrate mode:', mode);
                    
                    // Update description
                    if (mode === 'calculated') {{
                        description.textContent = 'Shows actual bitrate calculated from encoded video files';
                        note.innerHTML = '<strong>Note:</strong> VMAF values are plotted against actual achieved bitrates, not target bitrates. This automatically accounts for bitrate accuracy differences between devices.';
                        qualityData = qualityDataCalculated;
                    }} else {{
                        description.textContent = 'Shows target bitrate specified for encoding';
                        note.innerHTML = '<strong>Note:</strong> VMAF values are plotted against target bitrates specified for encoding. This shows the intended bitrate vs actual quality relationship.';
                        qualityData = qualityDataTarget;
                    }}
                    
                    // Update the plot dynamically
                    updateQualityPlots(mode);
                    
                    // Update current bitrate mode for BD-Rate analysis
                    currentBitrateMode = mode;
                    
                    // Extract raw quality data for BD-Rate analysis with new bitrate mode
                    if (typeof extractRawQualityData === 'function') {{
                        console.log('🔄 Extracting raw quality data with new bitrate mode:', mode);
                        extractRawQualityData();
                    }}
                    
                    // Regenerate BD-Rate analysis if it's already been initialized
                    if (typeof regenerateBDRateAnalysis === 'function' && rawQualityData) {{
                        var selectedReference = document.getElementById('bd-rate-reference-selector');
                        if (selectedReference && selectedReference.value) {{
                            console.log('🔄 Regenerating BD-Rate analysis with new bitrate mode:', mode);
                            regenerateBDRateAnalysis(selectedReference.value);
                        }}
                    }}
                }}
                
                function updateQualityPlots(mode) {{
                    console.log('Updating quality plots with mode:', mode);
                    var plotElement = document.getElementById('quality-plots');
                    if (plotElement && qualityData) {{
                        Plotly.react(plotElement, qualityData.data, qualityData.layout, {{responsive: true}});
                        console.log('Quality plots updated successfully');
                    }} else {{
                        console.log('Plot element or data not found');
                    }}
                }}
                
                function regenerateQualityPlots(mode) {{
                    if (!qualityDataBoth || !qualityDataBoth[mode]) {{
                        console.log('No data available for mode:', mode);
                        return;
                    }}
                    
                    var data = qualityDataBoth[mode];
                    var plotElement = document.getElementById('quality-plots');
                    
                    if (!plotElement) {{
                        console.log('Quality plot element not found');
                        return;
                    }}
                    
                    // Create new plot data
                    var newPlotData = {{
                        data: data.traces,
                        layout: qualityData.layout,
                        config: qualityData.config
                    }};
                    
                    // Update x-axis labels
                    newPlotData.layout.xaxis.title = data.bitrate_label;
                    newPlotData.layout.xaxis2.title = data.bitrate_label;
                    newPlotData.layout.xaxis3.title = data.bitrate_label;
                    
                    // Regenerate the plot
                    Plotly.react(plotElement, newPlotData.data, newPlotData.layout, newPlotData.config);
                    
                    console.log('Quality plots updated with', mode, 'bitrate mode');
                }}
                
                // Extract unique devices from quality data
                function extractDevices() {{
                    var devices = new Set();
                    console.log('🔍 EXTRACTING QUALITY DEVICES...');
                    // Use quality data for device extraction
                    var qualityDataForDevices = qualityData.data;
                    console.log('Quality data length:', qualityDataForDevices.length);
                    
                    qualityDataForDevices.forEach(function(trace, index) {{
                        var name = trace.name;
                        // Only process traces that have a name and match the expected format
                        if (name && typeof name === 'string') {{
                            // Extract device from trace name (format: "codec (device)")
                        var match = name.match(/\\(([^)]+)\\)/);
                        if (match) {{
                            devices.add(match[1]);
                                console.log('✅ Found quality device:', match[1]);
                            }}
                        }} else {{
                            console.log('❌ Skipping quality trace with invalid name:', name);
                        }}
                    }});
                    
                    var deviceArray = Array.from(devices).sort();
                    console.log('📊 QUALITY DEVICES:', deviceArray);
                    return deviceArray;
                }}
                
                // Extract unique codec-device pairs from quality data
                function extractQualityPairs() {{
                    var pairs = new Set();
                    console.log('🔍 EXTRACTING QUALITY PAIRS...');
                    // Use quality data for pair extraction
                    var qualityDataForPairs = qualityData.data;
                    console.log('Quality data length:', qualityDataForPairs.length);
                    
                    qualityDataForPairs.forEach(function(trace, index) {{
                        var name = trace.name;
                        // Only process traces that have a name and match the expected format
                        if (name && typeof name === 'string') {{
                            // Extract codec and device from trace name (format: "codec (device)")
                            var match = name.match(/([^(]+)\\(([^)]+)\\)/);
                            if (match) {{
                                var codec = match[1].trim();
                                var device = match[2].trim();
                                var pair = codec + ' (' + device + ')';
                                pairs.add(pair);
                                console.log('✅ Found quality pair:', pair);
                            }}
                        }} else {{
                            console.log('❌ Skipping quality trace with invalid name:', name);
                        }}
                    }});
                    
                    var pairArray = Array.from(pairs).sort();
                    console.log('📊 QUALITY PAIRS:', pairArray);
                    return pairArray;
                }}
                
                // Create device filter checkboxes
                function createDeviceFilters() {{
                    var devices = extractDevices();
                    var container = document.getElementById('device-filters');
                    devices.forEach(function(device) {{
                        var label = document.createElement('label');
                        label.innerHTML = '<input type="checkbox" id="filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_') + '" checked onchange="filterTraces()"> ' + device;
                        container.appendChild(label);
                        container.appendChild(document.createElement('br'));
                    }});
                }}
                
                // Extract unique devices from performance data
                function extractPerformanceDevices() {{
                    var devices = new Set();
                    console.log('🔍 EXTRACTING PERFORMANCE DEVICES...');
                    console.log('Performance data:', performanceData);
                    console.log('Performance data length:', performanceData.data.length);
                    
                    // Log all trace names first to see what we have
                    console.log('📋 ALL PERFORMANCE TRACE NAMES:');
                    performanceData.data.forEach(function(trace, index) {{
                        console.log('  Trace', index, ':', trace.name, '(type:', typeof trace.name, ')');
                    }});
                    
                    performanceData.data.forEach(function(trace, index) {{
                        try {{
                        var name = trace.name;
                            // Only process traces that have a name and match the expected format
                            if (name && typeof name === 'string' && name.length > 0) {{
                        // Extract device from trace name (format: "codec (device)")
                        var match = name.match(/\\(([^)]+)\\)/);
                        if (match) {{
                            devices.add(match[1]);
                                    console.log('✅ Found device:', match[1]);
                                }} else {{
                                    console.log('❌ No device match for:', name);
                                }}
                            }} else {{
                                console.log('❌ Skipping trace with invalid name:', name);
                            }}
                        }} catch (error) {{
                            console.log('❌ Error processing trace:', error, trace);
                        }}
                    }});
                    
                    var deviceArray = Array.from(devices).sort();
                    console.log('📊 FINAL EXTRACTED DEVICES:', deviceArray);
                    console.log('📊 EXPECTED DEVICES: V2413, SM-S936U1, Pixel 8');
                    return deviceArray;
                }}
                
                // Create performance device filter checkboxes
                function createPerformanceDeviceFilters() {{
                    console.log('🔧 CREATING PERFORMANCE DEVICE FILTERS...');
                    var devices = extractPerformanceDevices();
                    var container = document.getElementById('perf-device-filters');
                    console.log('Devices found:', devices);
                    console.log('Container element:', container);
                    if (devices.length === 0) {{
                        console.log('❌ No devices found for performance filters');
                        container.innerHTML = '<p>No device data available</p>';
                        return;
                    }}
                    console.log('✅ Creating', devices.length, 'device checkboxes');
                    devices.forEach(function(device) {{
                        var checkboxId = 'perf-filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_');
                        var label = document.createElement('label');
                        label.innerHTML = '<input type="checkbox" id="' + checkboxId + '" checked onchange="filterPerformanceTraces()"> ' + device;
                        container.appendChild(label);
                        container.appendChild(document.createElement('br'));
                        console.log('  Created checkbox:', checkboxId);
                    }});
                    console.log('🔧 PERFORMANCE DEVICE FILTERS CREATED');
                }}
                
                // Extract unique codec-device pairs from comparison data
                function extractComparisonDevices() {{
                    var pairs = new Set();
                    console.log('Comparison data:', comparisonData);
                    comparisonData.data.forEach(function(trace) {{
                        var name = trace.name;
                        console.log('Comparison trace name:', name);
                        // Only process traces that have a name and match the expected format
                        if (name && typeof name === 'string') {{
                            // Extract codec and device from trace name (format: "codec (device)")
                            var match = name.match(/([^(]+)\\(([^)]+)\\)/);
                            if (match) {{
                                var codec = match[1].trim();
                                var device = match[2].trim();
                                var pair = codec + ' (' + device + ')';
                                pairs.add(pair);
                                console.log('Found codec-device pair:', pair);
                            }}
                        }}
                    }});
                    console.log('Extracted comparison codec-device pairs:', Array.from(pairs));
                    return Array.from(pairs).sort();
                }}
                
                // Create comparison device filter checkboxes
                function createComparisonDeviceFilters() {{
                    var devices = extractComparisonDevices();
                    var container = document.getElementById('comp-device-filters');
                    console.log('Creating comparison device filters for devices:', devices);
                    console.log('Comparison container element:', container);
                    if (devices.length === 0) {{
                        console.log('No devices found for comparison filters');
                        container.innerHTML = '<p>No device data available</p>';
                        return;
                    }}
                    devices.forEach(function(device) {{
                        var label = document.createElement('label');
                        label.innerHTML = '<input type="checkbox" id="comp-filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_') + '" checked onchange="filterComparisonTraces()"> ' + device;
                        container.appendChild(label);
                        container.appendChild(document.createElement('br'));
                    }});
                }}
                
                // Create reference selector options
                function createReferenceSelector() {{
                    var pairs = extractComparisonDevices();
                    var selector = document.getElementById('reference-selector');
                    console.log('Creating reference selector for codec-device pairs:', pairs);
                    console.log('Reference selector element:', selector);
                    
                    if (!selector) {{
                        console.log('Reference selector element not found');
                        return;
                    }}
                    
                    // If no comparison pairs, try to get pairs from quality data
                    if (pairs.length === 0) {{
                        console.log('No comparison pairs found, trying quality data');
                        pairs = extractQualityPairs();
                        console.log('Quality pairs for reference:', pairs);
                    }}
                    
                    if (pairs.length === 0) {{
                        console.log('No codec-device pairs found for reference selector');
                        selector.innerHTML = '<option value="">No pairs available</option>';
                        return;
                    }}
                    
                    // Clear existing options except the first one
                    while (selector.children.length > 1) {{
                        selector.removeChild(selector.lastChild);
                    }}
                    
                    pairs.forEach(function(pair) {{
                        var option = document.createElement('option');
                        option.value = pair;
                        option.textContent = pair;
                        selector.appendChild(option);
                    }});
                    
                    console.log('Reference selector populated with', pairs.length, 'codec-device pairs');
                }}
                
                // Get codec type from trace name
                function getCodecType(traceName) {{
                    if (!traceName || typeof traceName !== 'string') {{
                        console.log('⚠️ getCodecType called with invalid name:', traceName);
                        return 'other';
                    }}
                    var name = traceName.toLowerCase();
                    if (name.includes('av1')) return 'av1';
                    if (name.includes('hevc') || name.includes('h265')) return 'hevc';
                    if (name.includes('avc') || name.includes('h264')) return 'avc';
                    if (name.includes('vp8')) return 'vp8';
                    return 'other';
                }}
                
                // Get device from trace name
                function getDevice(traceName) {{
                    if (!traceName || typeof traceName !== 'string') {{
                        console.log('⚠️ getDevice called with invalid name:', traceName);
                        return '';
                    }}
                    var match = traceName.match(/\\(([^)]+)\\)/);
                    return match ? match[1] : '';
                }}
                
                // Filter traces based on checkboxes
                function filterTraces() {{
                    var visibleTraces = [];
                    var codecFilters = {{
                        av1: document.getElementById('filter-av1').checked,
                        hevc: document.getElementById('filter-hevc').checked,
                        avc: document.getElementById('filter-avc').checked,
                        vp8: document.getElementById('filter-vp8').checked
                    }};
                    
                    // Check if any codec filters are enabled
                    var anyCodecEnabled = Object.values(codecFilters).some(function(enabled) {{ return enabled; }});
                    
                    qualityData.data.forEach(function(trace, index) {{
                        var codecType = getCodecType(trace.name);
                        var device = getDevice(trace.name);
                        var deviceFilterId = 'filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_');
                        var deviceCheckbox = document.getElementById(deviceFilterId);
                        var deviceEnabled = deviceCheckbox ? deviceCheckbox.checked : true;
                        
                        var codecEnabled = codecFilters[codecType] !== false;
                        
                        // If no codecs are selected, show nothing
                        if (!anyCodecEnabled) {{
                            return;
                        }}
                        
                        if (codecEnabled && deviceEnabled) {{
                            visibleTraces.push(index);
                        }}
                    }});
                    
                    // If no traces should be visible, hide all
                    if (visibleTraces.length === 0) {{
                        Plotly.restyle('quality-plots', {{visible: 'legendonly'}}, {{}});
                    }} else {{
                    Plotly.restyle('quality-plots', {{visible: 'legendonly'}}, {{}});
                    Plotly.restyle('quality-plots', {{visible: true}}, visibleTraces);
                    }}
                }}
                
                // Show all traces
                function showAllTraces() {{
                    // Check all codec filters
                    document.getElementById('filter-av1').checked = true;
                    document.getElementById('filter-hevc').checked = true;
                    document.getElementById('filter-avc').checked = true;
                    document.getElementById('filter-vp8').checked = true;
                    
                    // Check all device filters
                    var deviceCheckboxes = document.querySelectorAll('#device-filters input[type="checkbox"]');
                    deviceCheckboxes.forEach(function(checkbox) {{
                        checkbox.checked = true;
                    }});
                    
                    // Show all traces
                    Plotly.restyle('quality-plots', {{visible: true}}, {{}});
                }}
                
                // Hide all traces
                function hideAllTraces() {{
                    // Uncheck all codec filters
                    document.getElementById('filter-av1').checked = false;
                    document.getElementById('filter-hevc').checked = false;
                    document.getElementById('filter-avc').checked = false;
                    document.getElementById('filter-vp8').checked = false;
                    
                    // Uncheck all device filters
                    var deviceCheckboxes = document.querySelectorAll('#device-filters input[type="checkbox"]');
                    deviceCheckboxes.forEach(function(checkbox) {{
                        checkbox.checked = false;
                    }});
                    
                    // Hide all traces
                    Plotly.restyle('quality-plots', {{visible: 'legendonly'}}, {{}});
                }}
                
                // Filter performance traces based on checkboxes
                function filterPerformanceTraces() {{
                    console.log('🔍 Filtering performance traces...');
                    var visibleTraces = [];
                    var codecFilters = {{
                        av1: document.getElementById('perf-filter-av1').checked,
                        hevc: document.getElementById('perf-filter-hevc').checked,
                        avc: document.getElementById('perf-filter-avc').checked,
                        vp8: document.getElementById('perf-filter-vp8').checked
                    }};
                    
                    console.log('Codec filters:', codecFilters);
                    
                    // Check if any codec filters are enabled
                    var anyCodecEnabled = Object.values(codecFilters).some(function(enabled) {{ return enabled; }});
                    console.log('Any codec enabled:', anyCodecEnabled);
                    
                    // Debug: Check what device checkboxes exist
                    var deviceCheckboxes = document.querySelectorAll('#perf-device-filters input[type="checkbox"]');
                    console.log('🔍 DEVICE CHECKBOXES: Found', deviceCheckboxes.length, 'checkboxes');
                    if (deviceCheckboxes.length === 0) {{
                        console.log('❌ NO DEVICE CHECKBOXES FOUND! Check if createPerformanceDeviceFilters() is working');
                    }} else {{
                        deviceCheckboxes.forEach(function(checkbox, index) {{
                            console.log('  ', checkbox.id, '=', checkbox.checked);
                        }});
                    }}
                    
                    performanceData.data.forEach(function(trace, index) {{
                        // Skip traces with undefined or invalid names (likely confidence intervals)
                        if (!trace.name || typeof trace.name !== 'string' || trace.name.length === 0) {{
                            console.log('Trace', index, ': SKIPPING (undefined/invalid name)');
                            return;
                        }}
                        
                        var codecType = getCodecType(trace.name);
                        var device = getDevice(trace.name);
                        var deviceFilterId = 'perf-filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_');
                        var deviceCheckbox = document.getElementById(deviceFilterId);
                        var deviceEnabled = deviceCheckbox ? deviceCheckbox.checked : true;
                        
                        var codecEnabled = codecFilters[codecType] !== false;
                        
                        console.log('Trace', index, ':', trace.name, 'codec:', codecType, 'device:', device, 'codecEnabled:', codecEnabled, 'deviceEnabled:', deviceEnabled);
                        
                        // If no codecs are selected, show nothing
                        if (!anyCodecEnabled) {{
                            return;
                        }}
                        
                        if (codecEnabled && deviceEnabled) {{
                            visibleTraces.push(index);
                            console.log('  -> VISIBLE');
                        }} else {{
                            console.log('  -> HIDDEN');
                        }}
                    }});
                    
                    console.log('Visible traces:', visibleTraces);
                    
                    // If no traces should be visible, hide all
                    if (visibleTraces.length === 0) {{
                    Plotly.restyle('performance-plots', {{visible: 'legendonly'}}, {{}});
                    }} else {{
                        // First hide all traces
                        Plotly.restyle('performance-plots', {{visible: 'legendonly'}}, {{}});
                        
                        // Then show only the visible traces AND their associated confidence intervals
                        var tracesToShow = [];
                        visibleTraces.forEach(function(traceIndex) {{
                            var trace = performanceData.data[traceIndex];
                            if (trace && trace.legendgroup) {{
                                // Find all traces with the same legendgroup (main trace + confidence intervals)
                                performanceData.data.forEach(function(t, index) {{
                                    if (t.legendgroup === trace.legendgroup) {{
                                        tracesToShow.push(index);
                                    }}
                                }});
                            }} else {{
                                tracesToShow.push(traceIndex);
                            }}
                        }});
                        
                        console.log('Traces to show (including confidence intervals):', tracesToShow);
                        Plotly.restyle('performance-plots', {{visible: true}}, tracesToShow);
                    }}
                }}
                
                // Show all performance traces
                function showAllPerformanceTraces() {{
                    // Check all codec filters
                    document.getElementById('perf-filter-av1').checked = true;
                    document.getElementById('perf-filter-hevc').checked = true;
                    document.getElementById('perf-filter-avc').checked = true;
                    document.getElementById('perf-filter-vp8').checked = true;
                    
                    // Check all device filters
                    var deviceCheckboxes = document.querySelectorAll('#perf-device-filters input[type="checkbox"]');
                    deviceCheckboxes.forEach(function(checkbox) {{
                        checkbox.checked = true;
                    }});
                    
                    // Show all traces
                    Plotly.restyle('performance-plots', {{visible: true}}, {{}});
                }}
                
                // Hide all performance traces
                function hideAllPerformanceTraces() {{
                    // Uncheck all codec filters
                    document.getElementById('perf-filter-av1').checked = false;
                    document.getElementById('perf-filter-hevc').checked = false;
                    document.getElementById('perf-filter-avc').checked = false;
                    document.getElementById('perf-filter-vp8').checked = false;
                    
                    // Uncheck all device filters
                    var deviceCheckboxes = document.querySelectorAll('#perf-device-filters input[type="checkbox"]');
                    deviceCheckboxes.forEach(function(checkbox) {{
                        checkbox.checked = false;
                    }});
                    
                    // Hide all traces
                    Plotly.restyle('performance-plots', {{visible: 'legendonly'}}, {{}});
                }}
                
                // Filter comparison traces based on checkboxes
                function filterComparisonTraces() {{
                    var visibleTraces = [];
                    var codecFilters = {{
                        av1: document.getElementById('comp-filter-av1').checked,
                        hevc: document.getElementById('comp-filter-hevc').checked,
                        avc: document.getElementById('comp-filter-avc').checked,
                        vp8: document.getElementById('comp-filter-vp8').checked
                    }};
                    
                    // Check if any codec filters are enabled
                    var anyCodecEnabled = Object.values(codecFilters).some(function(enabled) {{ return enabled; }});
                    
                    comparisonData.data.forEach(function(trace, index) {{
                        var codecType = getCodecType(trace.name);
                        var device = getDevice(trace.name);
                        var deviceFilterId = 'comp-filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_');
                        var deviceCheckbox = document.getElementById(deviceFilterId);
                        var deviceEnabled = deviceCheckbox ? deviceCheckbox.checked : true;
                        
                        var codecEnabled = codecFilters[codecType] !== false;
                        
                        // If no codecs are selected, show nothing
                        if (!anyCodecEnabled) {{
                            return;
                        }}
                        
                        if (codecEnabled && deviceEnabled) {{
                            visibleTraces.push(index);
                        }}
                    }});
                    
                    // If no traces should be visible, hide all
                    if (visibleTraces.length === 0) {{
                        Plotly.restyle('comparison-plots', {{visible: 'legendonly'}}, {{}});
                    }} else {{
                        Plotly.restyle('comparison-plots', {{visible: 'legendonly'}}, {{}});
                        Plotly.restyle('comparison-plots', {{visible: true}}, visibleTraces);
                    }}
                }}
                
                // Show all comparison traces
                function showAllComparisonTraces() {{
                    // Check all codec filters
                    document.getElementById('comp-filter-av1').checked = true;
                    document.getElementById('comp-filter-hevc').checked = true;
                    document.getElementById('comp-filter-avc').checked = true;
                    document.getElementById('comp-filter-vp8').checked = true;
                    
                    // Check all device filters
                    var deviceCheckboxes = document.querySelectorAll('#comp-device-filters input[type="checkbox"]');
                    deviceCheckboxes.forEach(function(checkbox) {{
                        checkbox.checked = true;
                    }});
                    
                    // Show all traces
                    Plotly.restyle('comparison-plots', {{visible: true}}, {{}});
                }}
                
                // Hide all comparison traces
                function hideAllComparisonTraces() {{
                    // Uncheck all codec filters
                    document.getElementById('comp-filter-av1').checked = false;
                    document.getElementById('comp-filter-hevc').checked = false;
                    document.getElementById('comp-filter-avc').checked = false;
                    document.getElementById('comp-filter-vp8').checked = false;
                    
                    // Uncheck all device filters
                    var deviceCheckboxes = document.querySelectorAll('#comp-device-filters input[type="checkbox"]');
                    deviceCheckboxes.forEach(function(checkbox) {{
                        checkbox.checked = false;
                    }});
                    
                    // Hide all traces
                    Plotly.restyle('comparison-plots', {{visible: 'legendonly'}}, {{}});
                }}
                
                
                // Update reference selection
                function updateReference() {{
                    var selectedReference = document.getElementById('reference-selector').value;
                    console.log('🔄 updateReference called with:', selectedReference);
                    
                    if (selectedReference) {{
                        console.log('Reference codec-device pair selected:', selectedReference);
                        console.log('Available original VMAF delta data:', originalVmafDeltaData.length, 'traces');
                        console.log('Trace names:', originalVmafDeltaData.map(function(t) {{ return t.name; }}));
                        
                        // Regenerate comparison plots with new reference
                        regenerateComparisonPlots(selectedReference);
                    }} else {{
                        console.log('No reference selected');
                    }}
                }}
                
                // Store raw VMAF data for delta calculations
                var rawVmafData = {{}};
                
                // Store original plot data for bitrate scaling
                var originalQualityData = null;
                var originalComparisonData = null;
                
                // Store bitrate accuracy data for scaling calculations
                var bitrateAccuracyData = {{}};
                
                // Get scaled VMAF data based on bitrate accuracy
                function getScaledVmafData() {{
                    var scaledData = {{}};
                    
                    Object.keys(rawVmafData).forEach(function(device) {{
                        var deviceData = rawVmafData[device];
                        var scaledBitrates = [...deviceData.bitrates];
                        var scaledVmaf = [];
                        
                        if (bitrateAccuracyData[device]) {{
                            var accuracyData = bitrateAccuracyData[device];
                            
                            for (var i = 0; i < deviceData.bitrates.length; i++) {{
                                var requestedBitrate = deviceData.bitrates[i];
                                var actualBitrate = requestedBitrate; // Default fallback
                                var bitrateRatio = 1.0; // Default no scaling
                                
                                // Find closest requested bitrate match
                                var closestIndex = 0;
                                var minDiff = Math.abs(accuracyData.requestedBitrates[0] - requestedBitrate);
                                
                                for (var j = 1; j < accuracyData.requestedBitrates.length; j++) {{
                                    var diff = Math.abs(accuracyData.requestedBitrates[j] - requestedBitrate);
                                    if (diff < minDiff) {{
                                        minDiff = diff;
                                        closestIndex = j;
                                    }}
                                }}
                                
                                actualBitrate = accuracyData.actualBitrates[closestIndex];
                                bitrateRatio = requestedBitrate / actualBitrate;
                                
                                var scaledVmafValue = deviceData.vmaf[i] * bitrateRatio;
                                scaledVmaf.push(scaledVmafValue);
                            }}
                        }} else {{
                            // No bitrate accuracy data, use original values
                            scaledVmaf = [...deviceData.vmaf];
                        }}
                        
                        scaledData[device] = {{
                            bitrates: scaledBitrates,
                            vmaf: scaledVmaf,
                            name: deviceData.name
                        }};
                    }});
                    
                    console.log('Generated scaled VMAF data:', scaledData);
                    return scaledData;
                }}
                
                // Extract bitrate accuracy data from quality plots
                function extractBitrateAccuracyData() {{
                    bitrateAccuracyData = {{}};
                    if (qualityData && qualityData.data) {{
                        console.log('🔍 Looking for bitrate accuracy data in', qualityData.data.length, 'traces');
                        qualityData.data.forEach(function(trace, index) {{
                            console.log('Trace', index, ':', trace.name, 'xaxis:', trace.xaxis, 'yaxis:', trace.yaxis);
                            
                            if (trace.name && trace.x && trace.y && trace.x.length > 0 && trace.y.length > 0) {{
                                // Look for traces in the "Target vs Actual Bitrate" subplot (usually yaxis: 'y3' or similar)
                                var isBitrateAccuracy = (trace.xaxis === 'x3' && trace.yaxis === 'y3') || 
                                                      trace.name.toLowerCase().includes('bitrate') ||
                                                      (trace.x.every(function(x) {{ return x > 0 && x < 50000; }}) && 
                                                       trace.y.every(function(y) {{ return y > 0 && y < 50000; }}));
                                
                                console.log('  Bitrate accuracy check:', isBitrateAccuracy, 'x range:', Math.min(...trace.x), '-', Math.max(...trace.x), 'y range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                
                                if (isBitrateAccuracy) {{
                                    // Extract device name from trace name
                                    var deviceMatch = trace.name.match(/\\(([^)]+)\\)/);
                                    if (deviceMatch) {{
                                        var deviceName = deviceMatch[1];
                                        bitrateAccuracyData[deviceName] = {{
                                            requestedBitrates: trace.x, // Target bitrates
                                            actualBitrates: trace.y,    // Actual bitrates
                                            name: trace.name
                                        }};
                                        console.log('✅ Extracted bitrate accuracy for', deviceName, ':', trace.x.length, 'data points');
                                        console.log('  Sample data:', trace.x.slice(0, 3), '->', trace.y.slice(0, 3));
                                    }} else {{
                                        console.log('❌ Could not extract device name from:', trace.name);
                                    }}
                                }}
                            }}
                        }});
                    }}
                    console.log('📊 Final bitrate accuracy data:', bitrateAccuracyData);
                    console.log('📊 Available devices:', Object.keys(bitrateAccuracyData));
                }}
                
                // Toggle bitrate scaling for VMAF values
                function toggleBitrateScaling(tab) {{
                    console.log('🚨 FUNCTION CALLED: toggleBitrateScaling with tab:', tab);
                    var checkbox = document.getElementById('bitrate-scaling-' + tab);
                    var isEnabled = checkbox.checked;
                    
                    console.log('🔧 Bitrate scaling', isEnabled ? 'enabled' : 'disabled', 'for', tab, 'tab');
                    console.log('Checkbox element:', checkbox);
                    console.log('Data available - quality:', !!qualityData, 'comparison:', !!comparisonData);
                    
                    if (tab === 'quality') {{
                        if (isEnabled) {{
                            console.log('Applying scaling to quality tab...');
                            applyBitrateScaling('quality');
                        }} else {{
                            console.log('Restoring original quality data...');
                            restoreOriginalData('quality');
                        }}
                    }} else if (tab === 'comparison') {{
                        if (isEnabled) {{
                            console.log('Bitrate scaling enabled for comparison - regenerating with scaled data');
                        }} else {{
                            console.log('Bitrate scaling disabled for comparison - regenerating with original data');
                        }}
                        // Regenerate comparison plots with new scaling setting
                        var selectedReference = document.getElementById('reference-selector').value;
                        if (selectedReference) {{
                            regenerateComparisonPlots(selectedReference);
                        }}
                    }}
                }}
                
                // Apply bitrate scaling to VMAF values
                function applyBitrateScaling(tab) {{
                    console.log('🚀 Applying bitrate scaling to', tab, 'tab');
                    
                    if (tab === 'quality' && qualityData) {{
                        console.log('Quality data found, processing', qualityData.data.length, 'traces');
                        var scaledData = JSON.parse(JSON.stringify(qualityData)); // Deep copy
                        
                        scaledData.data.forEach(function(trace, index) {{
                            console.log('Processing trace', index, ':', trace.name, 'xaxis:', trace.xaxis, 'yaxis:', trace.yaxis);
                            if (trace.name && trace.x && trace.y && trace.x.length > 0 && trace.y.length > 0) {{
                                console.log('Original VMAF range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                
                                // Scale VMAF values by bitrate ratio
                                var scaledY = [];
                                var originalY = [...trace.y]; // Store original values
                                
                                // Extract device name from trace name
                                var deviceMatch = trace.name.match(/\\(([^)]+)\\)/);
                                var deviceName = deviceMatch ? deviceMatch[1] : null;
                                
                                for (var i = 0; i < trace.y.length; i++) {{
                                    var requestedBitrate = trace.x[i]; // VMAF plot x-axis is requested bitrate
                                    var actualBitrate = requestedBitrate; // Default fallback
                                    var bitrateRatio = 1.0; // Default no scaling
                                    
                                    // Find actual bitrate from bitrate accuracy data
                                    if (deviceName && bitrateAccuracyData[deviceName]) {{
                                        var accuracyData = bitrateAccuracyData[deviceName];
                                        
                                        // Find closest requested bitrate match
                                        var closestIndex = 0;
                                        var minDiff = Math.abs(accuracyData.requestedBitrates[0] - requestedBitrate);
                                        
                                        for (var j = 1; j < accuracyData.requestedBitrates.length; j++) {{
                                            var diff = Math.abs(accuracyData.requestedBitrates[j] - requestedBitrate);
                                            if (diff < minDiff) {{
                                                minDiff = diff;
                                                closestIndex = j;
                                            }}
                                        }}
                                        
                                        actualBitrate = accuracyData.actualBitrates[closestIndex];
                                        bitrateRatio = requestedBitrate / actualBitrate;
                                        
                                        console.log('Device:', deviceName, 'Requested:', requestedBitrate, 'Actual:', actualBitrate, 'Ratio:', bitrateRatio.toFixed(3));
                                    }} else {{
                                        console.log('No bitrate accuracy data found for device:', deviceName);
                                    }}
                                    
                                    var scaledVmaf = trace.y[i] * bitrateRatio;
                                    scaledY.push(scaledVmaf);
                                }}
                                trace.y = scaledY;
                                console.log('✅ Scaled trace:', trace.name, 'new range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                console.log('   Sample VMAF values (original -> scaled):');
                                for (var k = 0; k < Math.min(3, trace.y.length); k++) {{
                                    console.log('     ', originalY[k].toFixed(2), '->', scaledY[k].toFixed(2));
                                }}
                            }} else {{
                                console.log('❌ Skipping trace', index, '- invalid data');
                            }}
                        }});
                        
                        console.log('🔄 Updating quality plots...');
                        Plotly.react('quality-plots', scaledData.data, scaledData.layout, {{responsive: true}});
                        console.log('✅ Applied bitrate scaling to quality plots');
                        
                    }} else if (tab === 'comparison' && comparisonData) {{
                        console.log('Comparison data found, processing', comparisonData.data.length, 'traces');
                        var scaledData = JSON.parse(JSON.stringify(comparisonData)); // Deep copy
                        
                        scaledData.data.forEach(function(trace, index) {{
                            console.log('Processing comparison trace', index, ':', trace.name, 'xaxis:', trace.xaxis, 'yaxis:', trace.yaxis);
                            if (trace.xaxis === 'x' && trace.yaxis === 'y' && trace.name && trace.x && trace.y) {{
                                console.log('Original delta range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                
                                // Scale VMAF delta values by bitrate ratio
                                var scaledY = [];
                                var originalY = [...trace.y]; // Store original values
                                
                                // Extract device name from trace name
                                var deviceMatch = trace.name.match(/\\(([^)]+)\\)/);
                                var deviceName = deviceMatch ? deviceMatch[1] : null;
                                
                                for (var i = 0; i < trace.y.length; i++) {{
                                    var requestedBitrate = trace.x[i]; // Comparison plot x-axis is requested bitrate
                                    var actualBitrate = requestedBitrate; // Default fallback
                                    var bitrateRatio = 1.0; // Default no scaling
                                    
                                    // Find actual bitrate from bitrate accuracy data
                                    if (deviceName && bitrateAccuracyData[deviceName]) {{
                                        var accuracyData = bitrateAccuracyData[deviceName];
                                        
                                        // Find closest requested bitrate match
                                        var closestIndex = 0;
                                        var minDiff = Math.abs(accuracyData.requestedBitrates[0] - requestedBitrate);
                                        
                                        for (var j = 1; j < accuracyData.requestedBitrates.length; j++) {{
                                            var diff = Math.abs(accuracyData.requestedBitrates[j] - requestedBitrate);
                                            if (diff < minDiff) {{
                                                minDiff = diff;
                                                closestIndex = j;
                                            }}
                                        }}
                                        
                                        actualBitrate = accuracyData.actualBitrates[closestIndex];
                                        bitrateRatio = requestedBitrate / actualBitrate;
                                        
                                        console.log('Comparison Device:', deviceName, 'Requested:', requestedBitrate, 'Actual:', actualBitrate, 'Ratio:', bitrateRatio.toFixed(3));
                                    }} else {{
                                        console.log('No bitrate accuracy data found for comparison device:', deviceName);
                                    }}
                                    
                                    var scaledVmaf = trace.y[i] * bitrateRatio;
                                    scaledY.push(scaledVmaf);
                                }}
                                trace.y = scaledY;
                                console.log('✅ Scaled comparison trace:', trace.name, 'new range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                console.log('   Sample delta values (original -> scaled):');
                                for (var k = 0; k < Math.min(3, trace.y.length); k++) {{
                                    console.log('     ', originalY[k].toFixed(2), '->', scaledY[k].toFixed(2));
                                }}
                            }} else {{
                                console.log('❌ Skipping comparison trace', index, '- not a VMAF delta trace');
                            }}
                        }});
                        
                        console.log('🔄 Updating comparison plots...');
                        Plotly.react('comparison-plots', scaledData.data, scaledData.layout, {{responsive: true}});
                        console.log('✅ Applied bitrate scaling to comparison plots');
                    }} else {{
                        console.log('❌ No data available for', tab, 'tab');
                    }}
                }}
                
                // Restore original data without scaling
                function restoreOriginalData(tab) {{
                    console.log('Restoring original data for', tab, 'tab');
                    
                    if (tab === 'quality' && qualityData) {{
                        Plotly.react('quality-plots', qualityData.data, qualityData.layout, {{responsive: true}});
                        console.log('Restored original quality data');
                    }} else if (tab === 'comparison' && comparisonData) {{
                        // For comparison tab, we need to regenerate with current reference
                        var selectedReference = document.getElementById('reference-selector').value;
                        if (selectedReference) {{
                            regenerateComparisonPlots(selectedReference);
                        }} else {{
                            Plotly.react('comparison-plots', comparisonData.data, comparisonData.layout, {{responsive: true}});
                        }}
                        console.log('Restored original comparison data');
                    }} else {{
                        console.log('No data available to restore for', tab, 'tab');
                    }}
                }}
                
                // Extract and store raw VMAF data from quality plots
                function extractRawVmafData() {{
                    rawVmafData = {{}};
                    if (qualityData && qualityData.data) {{
                        qualityData.data.forEach(function(trace) {{
                            if (trace.name && trace.x && trace.y) {{
                                // Extract device from trace name
                                var match = trace.name.match(/\\(([^)]+)\\)/);
                                if (match) {{
                                    var device = match[1];
                                    if (!rawVmafData[device]) {{
                                        rawVmafData[device] = {{}};
                                    }}
                                    // Store bitrate (x) and VMAF (y) data
                                    rawVmafData[device].bitrates = trace.x;
                                    rawVmafData[device].vmaf = trace.y;
                                    rawVmafData[device].name = trace.name;
                                }}
                            }}
                        }});
                    }}
                    console.log('Extracted raw VMAF data:', rawVmafData);
                }}
                
                // Calculate VMAF deltas against reference device
                function calculateVmafDeltas(referenceDevice) {{
                    if (!rawVmafData[referenceDevice]) {{
                        console.log('No reference data found for:', referenceDevice);
                        return null;
                    }}
                    
                    var referenceBitrates = rawVmafData[referenceDevice].bitrates;
                    var referenceVmaf = rawVmafData[referenceDevice].vmaf;
                    var deltas = {{}};
                    
                    console.log('Reference device:', referenceDevice, 'bitrates:', referenceBitrates, 'vmaf:', referenceVmaf);
                    console.log('VMAF range check - min:', Math.min(...referenceVmaf), 'max:', Math.max(...referenceVmaf));
                    
                    // Calculate deltas for each device
                    Object.keys(rawVmafData).forEach(function(device) {{
                        console.log('Processing device:', device);
                        if (device === referenceDevice) {{
                            // Reference device should show as 0 delta
                            deltas[device] = {{
                                bitrates: referenceBitrates,
                                deltas: new Array(referenceBitrates.length).fill(0),
                                name: rawVmafData[device].name
                            }};
                            console.log('Reference device deltas:', deltas[device]);
                        }} else {{
                            var deviceBitrates = rawVmafData[device].bitrates;
                            var deviceVmaf = rawVmafData[device].vmaf;
                            var deltaVmaf = [];
                            
                            console.log('Device:', device, 'bitrates:', deviceBitrates, 'vmaf:', deviceVmaf);
                            
                            // Interpolate reference VMAF at device bitrates
                            for (var i = 0; i < deviceBitrates.length; i++) {{
                                var bitrate = deviceBitrates[i];
                                // Find closest reference bitrate
                                var closestRefIndex = 0;
                                var minDiff = Math.abs(referenceBitrates[0] - bitrate);
                                
                                for (var j = 1; j < referenceBitrates.length; j++) {{
                                    var diff = Math.abs(referenceBitrates[j] - bitrate);
                                    if (diff < minDiff) {{
                                        minDiff = diff;
                                        closestRefIndex = j;
                                    }}
                                }}
                                
                                // Calculate delta
                                var refVmaf = referenceVmaf[closestRefIndex];
                                var delta = deviceVmaf[i] - refVmaf;
                                deltaVmaf.push(delta);
                                
                                console.log('Bitrate:', bitrate, 'Device VMAF:', deviceVmaf[i], 'Ref VMAF:', refVmaf, 'Delta:', delta);
                            }}
                            
                            deltas[device] = {{
                                bitrates: deviceBitrates,
                                deltas: deltaVmaf,
                                name: rawVmafData[device].name
                            }};
                            console.log('Device deltas:', deltas[device]);
                        }}
                    }});
                    
                    console.log('Final deltas:', deltas);
                    return deltas;
                }}
                
                
                // Store original VMAF delta data for reference switching
                var originalVmafDeltaData = [];
                
                // Extract original VMAF delta data from comparison plots
                function extractOriginalVmafDeltaData() {{
                    originalVmafDeltaData = [];
                    console.log('🔍 extractOriginalVmafDeltaData called');
                    console.log('comparisonData:', comparisonData);
                    
                    if (comparisonData && comparisonData.data) {{
                        console.log('🔍 Extracting original VMAF delta data from', comparisonData.data.length, 'traces');
                        comparisonData.data.forEach(function(trace, index) {{
                            console.log('🔍 Trace', index, ':', trace.name, 'xaxis:', trace.xaxis, 'yaxis:', trace.yaxis, 'has x:', !!trace.x, 'has y:', !!trace.y);
                            // Look for VMAF delta traces - they should have data and be in the main plot
                            // Check if it's a VMAF delta trace by looking for traces with data and a name
                            if (trace.name && trace.x && trace.y && trace.x.length > 0 && trace.y.length > 0) {{
                                // Include all traces that have data - don't filter based on delta values
                                // The reference trace might show as 0 delta, which is valid
                                console.log('✅ VMAF delta trace', index, ':', trace.name, 'y range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                originalVmafDeltaData.push({{
                                    x: trace.x,
                                    y: trace.y,
                                    name: trace.name,
                                    mode: trace.mode || 'markers+lines',
                                    line: trace.line || {{ width: 2 }},
                                    marker: trace.marker || {{ size: 6 }},
                                    xaxis: trace.xaxis || 'x',
                                    yaxis: trace.yaxis || 'y'
                                }});
                            }} else {{
                                console.log('❌ Skipping trace', index, ':', trace.name, '- missing data or name');
                            }}
                        }});
                    }} else {{
                        console.log('❌ No comparison data available');
                    }}
                    console.log('📊 Extracted original VMAF delta data:', originalVmafDeltaData.length, 'traces');
                    console.log('Trace names:', originalVmafDeltaData.map(function(t) {{ return t.name; }}));
                }}
                
                // Regenerate comparison plots with selected reference
                function regenerateComparisonPlots(referencePair) {{
                    console.log('🔄 regenerateComparisonPlots called with:', referencePair);
                    console.log('📊 Available original VMAF delta data:', originalVmafDeltaData.length, 'traces');
                    
                    if (!referencePair) {{
                        console.log('❌ No reference codec-device pair selected');
                        return;
                    }}
                    
                    // Bitrate scaling is no longer available - VMAF deltas are already calculated against actual bitrates
                    var scalingEnabled = false;
                    console.log('Bitrate scaling disabled - using actual bitrates for delta calculation');
                    
                    // Find the reference trace in original VMAF delta data
                    var referenceTrace = null;
                    var referenceDeltas = [];
                    
                    console.log('🔍 Looking for reference pair:', referencePair, 'in', originalVmafDeltaData.length, 'traces');
                    console.log('Available trace names:', originalVmafDeltaData.map(function(t) {{ return t.name; }}));
                    
                    originalVmafDeltaData.forEach(function(trace, index) {{
                        console.log('🔍 Checking trace', index, ':', trace.name, 'matches', referencePair, '?', trace.name === referencePair);
                        if (trace.name && trace.name === referencePair) {{
                            referenceTrace = trace;
                            referenceDeltas = trace.y;
                            console.log('✅ Found reference trace:', trace.name, 'with deltas:', referenceDeltas);
                        }}
                    }});
                    
                    if (!referenceTrace) {{
                        console.log('❌ No reference trace found for:', referencePair);
                        console.log('Available traces:', originalVmafDeltaData.map(function(t) {{ return t.name; }}));
                        console.log('Trying to find partial match...');
                        
                        // Try partial matching as fallback
                        originalVmafDeltaData.forEach(function(trace, index) {{
                            if (trace.name && trace.name.includes(referencePair)) {{
                                console.log('🔍 Partial match found:', trace.name, 'contains', referencePair);
                                referenceTrace = trace;
                                referenceDeltas = trace.y;
                            }}
                        }});
                        
                        if (!referenceTrace) {{
                            console.log('❌ No partial match found either');
                            return;
                        }}
                    }}
                    
                    // Create new traces with adjusted deltas
                    var newTraces = [];
                    
                    originalVmafDeltaData.forEach(function(trace) {{
                        // Check if this is a valid trace with data
                        if (trace.name && trace.x && trace.y && trace.x.length > 0 && trace.y.length > 0) {{
                            var adjustedY = [];
                            
                            if (trace.name === referenceTrace.name) {{
                                // Reference device should show as 0 delta
                                adjustedY = new Array(trace.y.length).fill(0);
                                console.log('Setting reference device', trace.name, 'to 0 delta');
                            }} else {{
                                // Other devices: delta = device_delta - reference_delta
                                for (var i = 0; i < trace.y.length; i++) {{
                                    var deviceDelta = trace.y[i];
                                    var refDelta = referenceDeltas[i] || 0;
                                    var adjustedDelta = deviceDelta - refDelta;
                                    adjustedY.push(adjustedDelta);
                                }}
                                console.log('Adjusted deltas for', trace.name, ':', adjustedY.slice(0, 3), '...');
                            }}
                            
                            newTraces.push({{
                                x: trace.x,
                                y: adjustedY,
                                mode: trace.mode || 'markers+lines',
                                name: trace.name,
                                type: 'scatter',
                                line: trace.line || {{ width: 2 }},
                                marker: trace.marker || {{ size: 6 }},
                                xaxis: trace.xaxis || 'x',
                                yaxis: trace.yaxis || 'y',
                                showlegend: trace.showlegend !== false
                            }});
                        }} else {{
                            console.log('❌ Skipping invalid trace:', trace.name, 'x length:', trace.x ? trace.x.length : 0, 'y length:', trace.y ? trace.y.length : 0);
                        }}
                    }});
                    
                    // Update the plot with VMAF deltas
                    Plotly.react('comparison-plots', newTraces, comparisonData.layout, {{responsive: true}});
                    
                    // Update the plot title
                    var update = {{
                        'title': 'VMAF Delta Comparison (Reference: ' + referencePair + ')' + (scalingEnabled ? ' - Bitrate Scaled' : '')
                    }};
                    Plotly.relayout('comparison-plots', update);
                    
                    console.log('✅ REFERENCE SWITCHING ACTIVE: Updated comparison plot with', newTraces.length, 'traces');
                    console.log('Original traces:', originalVmafDeltaData.length, 'Processed traces:', newTraces.length);
                    console.log('Reference pair:', referencePair, 'now shows as 0 delta');
                    console.log('⚠️ Bitrate scaling for comparison not yet implemented - using original data');
                }}
                
                // Verify delta calculations using Quality tab VMAF data
                function verifyDeltaCalculations() {{
                    console.log('=== DELTA CALCULATION VERIFICATION ===');
                    
                    if (!qualityData || !originalVmafDeltaData) {{
                        console.log('❌ Missing data for verification');
                        return;
                    }}
                    
                    // Extract VMAF data from Quality tab plots
                    var qualityVmafData = {{}};
                    qualityData.data.forEach(function(trace) {{
                        if (trace.name && trace.x && trace.y && trace.xaxis === 'x' && trace.yaxis === 'y') {{
                            // This should be a VMAF trace from the first subplot
                            var deviceMatch = trace.name.match(/\\(([^)]+)\\)/);
                            if (deviceMatch) {{
                                var deviceName = deviceMatch[1];
                                
                                // Convert actual bitrates to target bitrates (round to nearest even number in Mbps)
                                var targetBitrates = trace.x.map(function(actualBitrate) {{
                                    // Convert kbps to Mbps and round to nearest even number
                                    var mbps = actualBitrate / 1000;
                                    var roundedMbps = Math.round(mbps / 2) * 2; // Round to nearest even number
                                    return roundedMbps * 1000; // Convert back to kbps
                                }});
                                
                                qualityVmafData[deviceName] = {{
                                    bitrates: targetBitrates, // Converted to target bitrates
                                    vmaf: trace.y,
                                    name: trace.name
                                }};
                                console.log('Found VMAF data for', deviceName, ':', trace.x.length, 'points');
                                console.log('  Actual bitrates:', trace.x.slice(0, 3), '-> Target bitrates:', targetBitrates.slice(0, 3));
                            }}
                        }}
                    }});
                    
                    console.log('Quality VMAF data devices:', Object.keys(qualityVmafData));
                    
                    if (Object.keys(qualityVmafData).length === 0) {{
                        console.log('❌ No VMAF data found in Quality tab');
                        return;
                    }}
                    
                    // Find a reference device
                    var referenceDevice = Object.keys(qualityVmafData)[0];
                    console.log('Using reference device:', referenceDevice);
                    
                    // Test with another device
                    var testDevice = null;
                    Object.keys(qualityVmafData).forEach(function(device) {{
                        if (device !== referenceDevice && !testDevice) {{
                            testDevice = device;
                        }}
                    }});
                    
                    if (!testDevice) {{
                        console.log('❌ No test device found');
                        return;
                    }}
                    
                    console.log('\\nTesting with device:', testDevice);
                    
                    // Calculate delta using Quality tab VMAF data
                    var referenceBitrates = qualityVmafData[referenceDevice].bitrates;
                    var referenceVmaf = qualityVmafData[referenceDevice].vmaf;
                    var deviceBitrates = qualityVmafData[testDevice].bitrates;
                    var deviceVmaf = qualityVmafData[testDevice].vmaf;
                    
                    console.log('Reference bitrates:', referenceBitrates.slice(0, 3), '...');
                    console.log('Reference VMAF:', referenceVmaf.slice(0, 3), '...');
                    console.log('Device bitrates:', deviceBitrates.slice(0, 3), '...');
                    console.log('Device VMAF:', deviceVmaf.slice(0, 3), '...');
                    
                    // Calculate first few deltas manually - match by target bitrate
                    var calculatedDeltas = [];
                    for (var i = 0; i < Math.min(3, deviceBitrates.length); i++) {{
                        var targetBitrate = deviceBitrates[i];
                        
                        // Find the reference VMAF at the same target bitrate
                        var refVmaf = null;
                        for (var j = 0; j < referenceBitrates.length; j++) {{
                            if (Math.abs(referenceBitrates[j] - targetBitrate) < 1) {{ // Exact match within 1 kbps
                                refVmaf = referenceVmaf[j];
                                break;
                            }}
                        }}
                        
                        if (refVmaf !== null) {{
                            var delta = deviceVmaf[i] - refVmaf;
                            calculatedDeltas.push(delta);
                            console.log('Point', i, ':', targetBitrate, 'kbps (exact match), VMAF', deviceVmaf[i], '-', refVmaf, '=', delta);
                        }} else {{
                            console.log('Point', i, ':', targetBitrate, 'kbps - no matching reference bitrate found');
                        }}
                    }}
                    
                    // Find original delta trace
                    var originalTrace = null;
                    originalVmafDeltaData.forEach(function(trace) {{
                        if (trace.name && trace.name.includes(testDevice)) {{
                            originalTrace = trace;
                        }}
                    }});
                    
                    if (originalTrace) {{
                        console.log('\\nOriginal deltas (first 3):', originalTrace.y.slice(0, 3));
                        console.log('Calculated deltas (first 3):', calculatedDeltas);
                        
                        var matches = true;
                        for (var i = 0; i < Math.min(3, calculatedDeltas.length); i++) {{
                            var diff = Math.abs(calculatedDeltas[i] - originalTrace.y[i]);
                            if (diff > 0.1) {{
                                matches = false;
                                console.log('MISMATCH at point', i, ':', calculatedDeltas[i], 'vs', originalTrace.y[i], 'diff:', diff);
                            }}
                        }}
                        
                        if (matches) {{
                            console.log('✅ CALCULATIONS MATCH!');
                        }} else {{
                            console.log('❌ CALCULATIONS DO NOT MATCH!');
                        }}
                    }} else {{
                        console.log('❌ No original delta trace found for', testDevice);
                    }}
                    
                    console.log('=== END VERIFICATION ===');
                }}
                
                // Initialize plots
                function initializePlots() {{
                    console.log('🚨 INITIALIZE PLOTS STARTED');
                    
                    // Load quality plots
                    Plotly.newPlot('quality-plots', qualityData.data, qualityData.layout, {{responsive: true}});
                    
                    // Load performance plots
                    Plotly.newPlot('performance-plots', performanceData.data, performanceData.layout, {{responsive: true}});
                    
                    // Load comparison plots
                    Plotly.newPlot('comparison-plots', comparisonData.data, comparisonData.layout, {{responsive: true}});
                    
                    // Extract raw VMAF data for delta calculations
                    extractRawVmafData();
                    
                    // Extract original VMAF delta data for reference switching
                    extractOriginalVmafDeltaData();
                    
                    // Extract bitrate accuracy data for scaling calculations
                    extractBitrateAccuracyData();
                    
                    // Verify delta calculations
                    console.log('🚨 CALLING VERIFICATION FUNCTION...');
                    verifyDeltaCalculations();
                    console.log('🚨 VERIFICATION FUNCTION COMPLETED');
                    
                    // Create device filters after plots are loaded
                    createDeviceFilters();
                    createPerformanceDeviceFilters();
                    createComparisonDeviceFilters();
                    createReferenceSelector();
                    if (typeof createBDRateReferenceSelector === 'function') {{
                        console.log('🔧 Creating BD-Rate reference selector');
                        createBDRateReferenceSelector();
                    }} else {{
                        console.log('❌ createBDRateReferenceSelector function not found');
                    }}
                    
                    // Extract BD-Rate data after quality plots are loaded (following comparison tab pattern)
                    if (typeof extractRawQualityData === 'function') {{
                        console.log('🔧 Extracting BD-Rate data');
                        extractRawQualityData();
                    }} else {{
                        console.log('❌ extractRawQualityData function not found');
                    }}
                    
                    // Apply initial filtering
                    filterTraces();
                    filterPerformanceTraces();
                    filterComparisonTraces();
                }}
                
                // Initialize when page loads
                window.onload = function() {{
                    initializePlots();
                }};
                
                // SI/TI Analysis Functions
                function initializeSITI() {{
                    console.log('🔄 SI/TI: Initializing...');
                    
                    // Render the initial plot
                    renderSITIPlot();
                }}
                
                function renderSITIPlot() {{
                    var container = document.getElementById('siti-plots');
                    if (!container) {{
                        console.log('❌ SI/TI: Plot container not found');
                        return;
                    }}
                    
                    if (!sitiData) {{
                        console.log('❌ SI/TI: No data available');
                        container.innerHTML = '<p>No SI/TI data available</p>';
                        return;
                    }}
                    
                    console.log('🔄 SI/TI: Rendering plot...');
                    
                    // Render the plot
                    Plotly.newPlot('siti-plots', sitiData.data, sitiData.layout, {{responsive: true}});
                    
                    console.log('✅ SI/TI: Plot rendered successfully');
                }}
            </script>
        </body>
        </html>
        """
        
        # Add BD-Rate section to HTML (will be populated dynamically)
        self.logger.info("Adding BD-Rate section to HTML for dynamic generation")
        available_codecs = self._get_available_codecs_from_results(test_results)
        self.logger.info(f"Available codecs for BD-Rate: {available_codecs}")
        # Always call the function, even with empty visualizations
        html_content = self.add_bd_rate_section_to_html(html_content, {}, test_results)
        
        # Save HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Interactive report saved to: {output_file}")
        return str(output_file)

    def create_bd_rate_visualizations(self, test_results: List[TestResult], 
                                    quality_metrics: List[str] = None,
                                    reference_codec: str = None,
                                    bitrate_mode: str = "calculated") -> Dict[str, go.Figure]:
        """
        Create BD-Rate visualizations for test results using the same data as quality plots
        
        Args:
            test_results: List of test results
            quality_metrics: List of quality metrics to visualize
            reference_codec: Reference codec for BD-Rate comparison (if None, uses pairwise comparison)
            
        Returns:
            Dictionary mapping visualization names to Plotly figures
        """
        # BD-Rate modules are required - will fail fast if not available
        
        if quality_metrics is None:
            quality_metrics = ['vmaf', 'psnr', 'ssim']
        
        visualizer = BDRateVisualizer()
        analyzer = AVABDRateAnalyzer()
        visualizations = {}
        
        # Use the same data source as quality plots - read from test_results
        all_data = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'quality_csv' in result.test_data:
                csv_file = result.test_data['quality_csv']
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            # Add device info to the dataframe
                            df['device_serial'] = result.device_serial
                            df['test_name'] = result.test_name
                            
                            # Apply custom labels if this is a custom labeled test
                            df = self._apply_custom_labels_to_dataframe(df, result)
                            
                            all_data.append(df)
                    except pd.errors.EmptyDataError:
                        self.logger.warning(f"Skipping empty quality CSV file: {csv_file}")
                        continue
        
        if not all_data:
            self.logger.warning("No quality data available for BD-Rate analysis")
            return {}
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Determine which bitrate column to use
        if bitrate_mode == "target":
            bitrate_col = "bitrate_bps"
        else:  # calculated
            bitrate_col = "calculated_bitrate_bps"
        
        # Check if we have the required columns
        required_columns = ['vmaf_mean', bitrate_col, 'codec']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            self.logger.warning(f"Missing required columns for BD-Rate analysis: {missing_columns}")
            return {}
        
        # Create a temporary CSV file with the combined data for BD-Rate analysis
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        combined_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        
        try:
            # Get available codecs for reference selection
            available_codecs = analyzer.get_available_codecs(temp_csv.name)
            
            # If no reference codec specified, use the first one as default
            if reference_codec is None and available_codecs:
                reference_codec = available_codecs[0]
            
            # First pass: collect all BD-Rate results to determine consistent scaling
            all_bd_results = {}
            all_rd_curves = {}
            
            for metric in quality_metrics:
                try:
                    # Get RD curves for this metric
                    rd_curves_metric = analyzer.extract_rd_curves_from_csv(temp_csv.name, metric)
                    
                    if len(rd_curves_metric) < 2:
                        self.logger.warning(f"Insufficient codecs for BD-Rate comparison ({metric})")
                        continue
                    
                    # Calculate BD-Rate results
                    if reference_codec and reference_codec in available_codecs:
                        # Use reference-based comparison
                        bd_results = analyzer.compare_codecs_against_reference(temp_csv.name, reference_codec, metric)
                        viz_key = f"combined_{metric}_ref_{reference_codec.replace('.', '_')}"
                    else:
                        # Use pairwise comparison
                        bd_results = analyzer.compare_all_codecs_from_csv(temp_csv.name, metric)
                        viz_key = f"combined_{metric}"
                    
                    if not bd_results:
                        self.logger.warning(f"No BD-Rate results for {metric}")
                        continue
                    
                    all_bd_results[metric] = bd_results
                    all_rd_curves[metric] = rd_curves_metric
                    
                except Exception as e:
                    self.logger.error(f"Failed to collect BD-Rate data for {metric}: {e}")
            
            # Calculate consistent y-axis range for all BD-Rate bar charts
            all_bd_rates = []
            for bd_results in all_bd_results.values():
                for result in bd_results.values():
                    all_bd_rates.append(result.bd_rate)
            
            if all_bd_rates:
                # Add some padding to the range
                min_rate = min(all_bd_rates)
                max_rate = max(all_bd_rates)
                padding = (max_rate - min_rate) * 0.1  # 10% padding
                y_range = [min_rate - padding, max_rate + padding]
            else:
                y_range = None
            
            # Second pass: create visualizations with consistent scaling
            for metric in quality_metrics:
                if metric not in all_bd_results:
                    continue
                    
                try:
                    bd_results = all_bd_results[metric]
                    rd_curves_metric = all_rd_curves[metric]
                    
                    if reference_codec and reference_codec in available_codecs:
                        viz_key = f"combined_{metric}_ref_{reference_codec.replace('.', '_')}"
                    else:
                        viz_key = f"combined_{metric}"
                    
                    # Rate-distortion curves
                    rd_fig = visualizer.create_rd_curve_plot(rd_curves_metric, metric)
                    visualizations[f"{viz_key}_rd_curves"] = rd_fig
                    
                    # BD-Rate bar chart with consistent scaling
                    if reference_codec and reference_codec in available_codecs:
                        bd_fig = visualizer.create_reference_bd_rate_chart(bd_results, reference_codec, metric)
                    else:
                        bd_fig = visualizer.create_bd_rate_bar_chart(bd_results, metric)
                    
                    # Apply consistent y-axis range if available
                    if y_range is not None:
                        bd_fig.update_layout(yaxis=dict(range=y_range, zeroline=True, zerolinecolor='black', zerolinewidth=2))
                    
                    visualizations[f"{viz_key}_bd_rate_bars"] = bd_fig
                    
                    # Comprehensive comparison
                    comp_fig = visualizer.create_comprehensive_comparison(rd_curves_metric, bd_results, metric)
                    visualizations[f"{viz_key}_comprehensive"] = comp_fig
                    
                except Exception as e:
                    self.logger.error(f"Failed to create visualization for {metric}: {e}")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_csv.name)
        
        return visualizations

    def add_bd_rate_section_to_html(self, html_content: str, bd_visualizations: Dict[str, go.Figure], test_results: List[TestResult] = None) -> str:
        """
        Add BD-Rate visualization section to HTML report
        
        Args:
            html_content: Existing HTML content
            bd_visualizations: Dictionary of BD-Rate visualizations
            
        Returns:
            Updated HTML content with BD-Rate section
        """
        # Always add BD-Rate section (visualizations are generated dynamically)
        
        # Find the BD-Rate content div
        bd_rate_content_start = html_content.find('<div id="bd-rate-content">')
        bd_rate_content_end = html_content.find('</div>', bd_rate_content_start)
        
        self.logger.info(f"BD-Rate content div search: start={bd_rate_content_start}, end={bd_rate_content_end}")
        
        if bd_rate_content_start == -1 or bd_rate_content_end == -1:
            self.logger.warning("Could not find BD-Rate content div in HTML")
            return html_content
        
        # Create BD-Rate section content
        # Extract available codecs from all test results (handles custom labels)
        available_codecs = self._get_available_codecs_from_results(test_results) if test_results else []
        
        bd_section_content = self._create_bd_rate_html_content(bd_visualizations, available_codecs)
        
        # Add JavaScript for BD-Rate reference selector (always include the functions)
        bd_rate_js = f"""
            <script>
            // Create BD-Rate reference selector options
            function createBDRateReferenceSelector() {{
                var codecs = {json.dumps(available_codecs)};
                var selector = document.getElementById('bd-rate-reference-selector');
                console.log('Creating BD-Rate reference selector for codecs:', codecs);
                console.log('BD-Rate reference selector element:', selector);
                
                if (!selector) {{
                    console.log('BD-Rate reference selector element not found');
                    return;
                }}
                
                if (codecs.length === 0) {{
                    console.log('No codecs found for BD-Rate reference selector');
                    selector.innerHTML = '<option value="">No codecs available</option>';
                    return;
                }}
                
                // Clear existing options except the first one
                selector.innerHTML = '<option value="">Select Reference...</option>';
                
                // Add codec options
                codecs.forEach(function(codec) {{
                    var option = document.createElement('option');
                    option.value = codec;
                    option.textContent = codec;
                    selector.appendChild(option);
                }});
                
                console.log('BD-Rate reference selector populated with', codecs.length, 'codecs');
            }}
            
            // Store raw quality data for BD-Rate calculations
            var rawQualityData = {{}};
            
            // Update BD-Rate reference selection
            function updateBDRateReference() {{
                console.log('🔄 updateBDRateReference called');
                
                // Ensure data is extracted first
                if (Object.keys(rawQualityData).length === 0) {{
                    console.log('🔧 No data available, extracting now...');
                    if (typeof extractRawQualityData === 'function') {{
                        extractRawQualityData();
                    }}
                }}
                
                var selectedReference = document.getElementById('bd-rate-reference-selector').value;
                console.log('BD-Rate reference selected:', selectedReference);
                
                if (selectedReference) {{
                    console.log('✅ Reference selected, calling regenerateBDRateAnalysis');
                    // Regenerate BD-Rate visualizations with new reference
                    regenerateBDRateAnalysis(selectedReference);
                }} else {{
                    console.log('❌ No reference selected');
                }}
            }}
            
            // Extract raw quality data from quality plots (following comparison tab pattern)
            function extractRawQualityData() {{
                console.log('🔍 Starting BD-Rate data extraction...');
                rawQualityData = {{}};
                
                if (!qualityData || !qualityData.data) {{
                    console.log('❌ No qualityData available');
                    return;
                }}
                
                console.log('📊 Processing', qualityData.data.length, 'traces for BD-Rate');
                
                qualityData.data.forEach(function(trace) {{
                    if (trace.name && trace.x && trace.y) {{
                        // Extract codec and device from trace name: "codec (device)"
                        var match = trace.name.match(/([^(]+)\\(([^)]+)\\)/);
                        if (match) {{
                            var codec = match[1].trim();
                            var device = match[2].trim();
                            
                            if (!rawQualityData[codec]) {{
                                rawQualityData[codec] = {{}};
                            }}
                            
                            // Determine quality metric based on subplot position or trace characteristics
                            var metric = 'unknown';
                            
                            // Check yaxis property first
                            if (trace.yaxis === 'y') {{
                                metric = 'vmaf';
                            }} else if (trace.yaxis === 'y2') {{
                                metric = 'psnr';
                            }} else if (trace.yaxis === 'y3') {{
                                metric = 'ssim';
                            }} else {{
                                // Fallback: determine metric based on data characteristics
                                var yValues = trace.y;
                                if (yValues && yValues.length > 0) {{
                                    var minY = Math.min(...yValues);
                                    var maxY = Math.max(...yValues);
                                    
                                    // VMAF: typically 0-100
                                    if (minY >= 0 && maxY <= 100) {{
                                        metric = 'vmaf';
                                    }}
                                    // PSNR: typically 20-50 dB
                                    else if (minY >= 15 && maxY <= 60) {{
                                        metric = 'psnr';
                                    }}
                                    // SSIM: typically 0-1
                                    else if (minY >= 0 && maxY <= 1) {{
                                        metric = 'ssim';
                                    }}
                                }}
                            }}
                            
                            if (metric !== 'unknown') {{
                                console.log('✅ BD-Rate: Found', metric, 'data for', codec, ':', trace.x.length, 'points');
                                rawQualityData[codec][metric] = {{
                                    bitrates: trace.x,
                                    qualities: trace.y,
                                    device: device,
                                    name: trace.name
                                }};
                            }} else {{
                                console.log('❌ BD-Rate: Unknown metric for', codec, ':', trace.name, 'y range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                            }}
                        }}
                    }}
                }});
                
                console.log('📊 Extracted BD-Rate data for codecs:', Object.keys(rawQualityData));
            }}
            
            // Regenerate BD-Rate analysis with new reference (following comparison tab pattern)
            function regenerateBDRateAnalysis(referenceCodec) {{
                console.log('🔄 Regenerating BD-Rate analysis with reference:', referenceCodec);
                
                if (!referenceCodec) {{
                    console.log('❌ No reference codec selected');
                    return;
                }}
                
                if (!rawQualityData[referenceCodec]) {{
                    console.log('❌ No data found for reference codec:', referenceCodec);
                    return;
                }}
                
                // Show all metric sections
                document.getElementById('bd-rate-vmaf').style.display = 'block';
                document.getElementById('bd-rate-psnr').style.display = 'block';
                document.getElementById('bd-rate-ssim').style.display = 'block';
                
                // Generate BD-Rate analysis for each metric
                generateBDRateAnalysis('vmaf', referenceCodec);
                generateBDRateAnalysis('psnr', referenceCodec);
                generateBDRateAnalysis('ssim', referenceCodec);
            }}
            
            // Generate BD-Rate analysis for a specific metric
            function generateBDRateAnalysis(metric, referenceCodec) {{
                console.log('📊 Generating BD-Rate analysis for', metric, 'with reference', referenceCodec);
                
                // Get reference data
                var referenceData = rawQualityData[referenceCodec][metric];
                if (!referenceData) {{
                    console.log('❌ No reference data for', metric);
                    return;
                }}
                
                // Collect all codec data for this metric
                var allCodecData = [];
                Object.keys(rawQualityData).forEach(function(codec) {{
                    if (rawQualityData[codec][metric]) {{
                        allCodecData.push({{
                            codec: codec,
                            bitrates: rawQualityData[codec][metric].bitrates,
                            qualities: rawQualityData[codec][metric].qualities
                        }});
                    }}
                }});
                
                if (allCodecData.length < 2) {{
                    console.log('❌ Not enough codecs for BD-Rate analysis');
                    return;
                }}
                
                // Create rate-distortion curves plot
                createRDCurvesPlot(metric, allCodecData);
                
                // Create BD-Rate comparison plot
                createBDRateComparisonPlot(metric, allCodecData, referenceCodec);
            }}
            
            // Create rate-distortion curves plot
            function createRDCurvesPlot(metric, codecData) {{
                var traces = [];
                
                codecData.forEach(function(data) {{
                    traces.push({{
                        x: data.bitrates,
                        y: data.qualities,
                        mode: 'lines+markers',
                        name: data.codec,
                        line: {{ width: 3 }},
                        marker: {{ size: 6 }}
                    }});
                }});
                
                var layout = {{
                    title: metric.toUpperCase() + ' Rate-Distortion Curves',
                    xaxis: {{ 
                        title: 'Bitrate (kbps)',
                        type: 'log',
                        showgrid: true
                    }},
                    yaxis: {{ 
                        title: metric.toUpperCase(),
                        showgrid: true
                    }},
                    width: 800,
                    height: 600,
                    margin: {{ t: 60, b: 60, l: 80, r: 40 }},
                    showlegend: true
                }};
                
                var plotDiv = 'bd-rate-' + metric + '-rd';
                Plotly.newPlot(plotDiv, traces, layout, {{responsive: true}});
            }}
            
            // Create BD-Rate comparison plot
            function createBDRateComparisonPlot(metric, codecData, referenceCodec) {{
                // Calculate BD-Rate for each codec vs reference
                var bdRates = [];
                var codecNames = [];
                
                codecData.forEach(function(data) {{
                    if (data.codec !== referenceCodec) {{
                        // Simple BD-Rate calculation (placeholder)
                        // In a full implementation, this would use proper cubic spline interpolation
                        var bdRate = calculateSimpleBDRate(data, codecData.find(c => c.codec === referenceCodec));
                        bdRates.push(bdRate);
                        codecNames.push(data.codec);
                    }}
                }});
                
                var colors = bdRates.map(function(rate) {{
                    return rate < 0 ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
                }});
                
                var trace = {{
                    x: codecNames,
                    y: bdRates,
                    type: 'bar',
                    marker: {{ color: colors }},
                    text: bdRates.map(function(rate) {{
                        return rate.toFixed(2) + '%';
                    }}),
                    textposition: 'auto'
                }};
                
                var layout = {{
                    title: metric.toUpperCase() + ' BD-Rate vs ' + referenceCodec,
                    xaxis: {{ title: 'Codec' }},
                    yaxis: {{ 
                        title: 'BD-Rate (%)',
                        zeroline: true,
                        zerolinecolor: 'black'
                    }},
                    width: 800,
                    height: 600,
                    margin: {{ t: 60, b: 60, l: 80, r: 40 }},
                    showlegend: false
                }};
                
                var plotDiv = 'bd-rate-' + metric + '-bd';
                Plotly.newPlot(plotDiv, [trace], layout, {{responsive: true}});
            }}
            
            // Simple BD-Rate calculation (placeholder)
            function calculateSimpleBDRate(codecData, referenceData) {{
                // This is a simplified calculation for demonstration
                // A full implementation would use proper cubic spline interpolation
                var avgCodecQuality = codecData.qualities.reduce((a, b) => a + b, 0) / codecData.qualities.length;
                var avgReferenceQuality = referenceData.qualities.reduce((a, b) => a + b, 0) / referenceData.qualities.length;
                var avgCodecBitrate = codecData.bitrates.reduce((a, b) => a + b, 0) / codecData.bitrates.length;
                var avgReferenceBitrate = referenceData.bitrates.reduce((a, b) => a + b, 0) / referenceData.bitrates.length;
                
                // Simple percentage difference
                var bitrateDiff = ((avgCodecBitrate - avgReferenceBitrate) / avgReferenceBitrate) * 100;
                return bitrateDiff;
            }}
            
            // Auto-select first codec when page loads (following comparison tab pattern)
            window.addEventListener('load', function() {{
                console.log('🔄 BD-Rate: Page loaded, starting initialization...');
                
                setTimeout(function() {{
                    console.log('🔄 BD-Rate: Checking for reference selector...');
                    var selector = document.getElementById('bd-rate-reference-selector');
                    console.log('🔄 BD-Rate: Selector found:', selector);
                    
                    if (selector && selector.options.length > 1) {{
                        console.log('🔄 BD-Rate: Auto-selecting first codec:', selector.options[1].value);
                        // Select the first actual codec as default
                        selector.value = selector.options[1].value;
                        updateBDRateReference();
                    }} else {{
                        console.log('❌ BD-Rate: No selector or options found');
                    }}
                }}, 1000); // Give time for data extraction
            }});
            
            // Also try to initialize when BD-Rate tab is clicked
            function initializeBDRate() {{
                console.log('🔄 BD-Rate: Manual initialization called...');
                
                // Hide manual init button
                var manualInit = document.getElementById('bd-rate-manual-init');
                if (manualInit) {{
                    manualInit.style.display = 'none';
                }}
                
                // First, populate the dropdown
                if (typeof createBDRateReferenceSelector === 'function') {{
                    console.log('🔧 BD-Rate: Populating dropdown...');
                    createBDRateReferenceSelector();
                }}
                
                // Ensure data is extracted
                if (Object.keys(rawQualityData).length === 0) {{
                    console.log('🔧 BD-Rate: No data, extracting now...');
                    if (typeof extractRawQualityData === 'function') {{
                        extractRawQualityData();
                    }}
                }}
                
                // Try to select first codec
                var selector = document.getElementById('bd-rate-reference-selector');
                if (selector && selector.options.length > 1) {{
                    if (selector.value === '') {{
                        console.log('🔄 BD-Rate: Selecting first codec:', selector.options[1].value);
                        selector.value = selector.options[1].value;
                    }}
                    updateBDRateReference();
                }} else {{
                    console.log('❌ BD-Rate: No selector or options found');
                    if (manualInit) {{
                        manualInit.style.display = 'block';
                    }}
                }}
            }}
            </script>
            """
        bd_section_content += bd_rate_js
        
        # Replace the placeholder content
        updated_html = (html_content[:bd_rate_content_start + len('<div id="bd-rate-content">')] + 
                       bd_section_content + 
                       html_content[bd_rate_content_end:])
        
        return updated_html

    def _apply_custom_labels_to_dataframe(self, df: pd.DataFrame, result: TestResult) -> pd.DataFrame:
        """Apply custom labels to dataframe if this is a custom labeled test"""
        if result.test_name.startswith("quality_analysis_") and len(result.test_name.split("_")) > 2:
            custom_label = "_".join(result.test_name.split("_")[2:])  # Everything after "quality_analysis_"
            if 'codec' in df.columns:
                df['codec'] = custom_label
        
        # Add reference file information from test_data
        if result.test_data and 'sourcefile' in result.test_data:
            df['reference_file'] = result.test_data['sourcefile']
        else:
            df['reference_file'] = 'unknown'
            
        return df

    def _get_available_codecs_from_results(self, test_results: List[TestResult]) -> List[str]:
        """Extract available codecs from test results"""
        available_codecs = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'quality_csv' in result.test_data:
                # Check if this is a custom labeled test (quality_analysis_<label>)
                if result.test_name.startswith("quality_analysis_") and len(result.test_name.split("_")) > 2:
                    # Extract the custom label from test_name
                    custom_label = "_".join(result.test_name.split("_")[2:])  # Everything after "quality_analysis_"
                    available_codecs.append(custom_label)
                else:
                    # Original behavior: extract codecs from CSV
                    csv_file = result.test_data['quality_csv']
                    if os.path.exists(csv_file):
                        try:
                            df = pd.read_csv(csv_file)
                            if 'codec' in df.columns:
                                codecs_in_file = df['codec'].unique().tolist()
                                available_codecs.extend(codecs_in_file)
                        except Exception as e:
                            self.logger.warning(f"Failed to read CSV for codec extraction: {e}")
        
        # Remove duplicates and sort
        return sorted(list(set(available_codecs)))

    def _create_bd_rate_html_content(self, bd_visualizations: Dict[str, go.Figure], 
                                   available_codecs: List[str] = None) -> str:
        """Create HTML content for BD-Rate visualizations within the tab"""
        
        # For dynamic BD-Rate, we'll generate the plots in JavaScript
        # Just create placeholder sections that will be populated dynamically
        
        # Create reference codec dropdown if codecs are available
        reference_dropdown = ""
        if available_codecs and len(available_codecs) > 1:
            reference_dropdown = f"""
            <div class="reference-selector">
                <h4>Reference Selection:</h4>
                <select id="bd-rate-reference-selector" onchange="updateBDRateReference()">
                    <option value="">Select Reference...</option>
                </select>
            </div>
            """
        
        html_content = f"""
        <div class="bd-rate-section">
            <h2>Bjøntegaard-Delta (BD-Rate) Analysis</h2>
            <p>BD-Rate measures the average percentage bit rate difference between two rate-distortion curves at equal measured distortion. Negative values indicate better compression efficiency (lower bitrate for same quality).</p>
            {reference_dropdown}
            
            <div id="bd-rate-content">
                <p>Select a reference codec above to view BD-Rate analysis.</p>
                <div id="bd-rate-loading" style="display: none;">
                    <p>Loading BD-Rate analysis...</p>
                </div>
                <div id="bd-rate-manual-init" style="display: none;">
                    <p>BD-Rate analysis not loaded. <button onclick="initializeBDRate()">Click here to initialize</button></p>
                </div>
                
                <div id="bd-rate-vmaf" class="bd-rate-metric-section" style="display: none;">
                    <h3>VMAF Analysis</h3>
                    <div id="bd-rate-vmaf-rd" class="bd-rate-plot"></div>
                    <div id="bd-rate-vmaf-bd" class="bd-rate-plot"></div>
                </div>
                
                <div id="bd-rate-psnr" class="bd-rate-metric-section" style="display: none;">
                    <h3>PSNR Analysis</h3>
                    <div id="bd-rate-psnr-rd" class="bd-rate-plot"></div>
                    <div id="bd-rate-psnr-bd" class="bd-rate-plot"></div>
                </div>
                
                <div id="bd-rate-ssim" class="bd-rate-metric-section" style="display: none;">
                    <h3>SSIM Analysis</h3>
                    <div id="bd-rate-ssim-rd" class="bd-rate-plot"></div>
                    <div id="bd-rate-ssim-bd" class="bd-rate-plot"></div>
                </div>
            </div>
        </div>
        """
        
        return html_content

    def generate_csv_only_report(self, quality_csv_files: List[str], 
                                quality_labels: List[str] = None,
                                stats_csv_files: List[str] = None,
                                bitrate_mode: str = "calculated") -> Dict[str, str]:
        """Generate reports directly from CSV files without requiring quality folders
        
        Args:
            quality_csv_files: List of paths to quality CSV files
            quality_labels: Optional list of labels for the quality CSV files
            stats_csv_files: Optional list of paths to stats CSV files
            bitrate_mode: "calculated" for calculated_bitrate_bps or "target" for bitrate_bps
            
        Returns:
            Dictionary with report type and file path
        """
        # Validate input files
        for csv_file in quality_csv_files:
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Quality CSV file not found: {csv_file}")
        
        if stats_csv_files:
            for csv_file in stats_csv_files:
                if not os.path.exists(csv_file):
                    raise FileNotFoundError(f"Stats CSV file not found: {csv_file}")
        
        # Create TestResult objects from CSV files
        test_results = []
        
        for i, quality_csv in enumerate(quality_csv_files):
            # Create test data dictionary
            test_data = {"quality_csv": quality_csv}
            
            # Add stats CSV if provided
            if stats_csv_files and i < len(stats_csv_files):
                test_data["stats_csv"] = [stats_csv_files[i]]
            
            # Extract test info from filename or CSV content
            test_name = "quality_analysis"
            device_serial = "unknown"
            
            # Use custom label if provided
            if quality_labels and i < len(quality_labels) and quality_labels[i]:
                test_name = f"quality_analysis_{quality_labels[i]}"
            else:
                # Try to extract info from CSV content
                try:
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
        
        # Generate the interactive report
        interactive_report = self.generate_interactive_report(test_results, None, bitrate_mode)
        return {"interactive": interactive_report}

    def generate_comprehensive_report(self, test_results: List[TestResult], 
                                    stats_files: List[str] = None, bitrate_mode: str = "calculated") -> Dict[str, str]:
        """Generate comprehensive reports - wrapper for interactive report
        
        Args:
            test_results: List of test results
            stats_files: Optional list of stats files
            bitrate_mode: "calculated" for calculated_bitrate_bps or "target" for bitrate_bps
        """
        interactive_report = self.generate_interactive_report(test_results, stats_files, bitrate_mode)
        return {"interactive": interactive_report}
