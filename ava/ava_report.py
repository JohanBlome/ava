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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo


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
    
    def create_quality_plots_from_csv(self, test_results: List[TestResult]) -> go.Figure:
        """Create quality plots directly from CSV files"""
            
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
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        # Add device info to the dataframe
                        df['device_serial'] = result.device_serial
                        df['test_name'] = result.test_name
                        all_data.append(df)
        
        if not all_data:
            return fig
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # VMAF plot - use calculated_bitrate_bps for X-axis
        if 'vmaf_mean' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec].sort_values('calculated_bitrate_bps')
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Get consistent color and line style
                    color = self._get_device_color(model)
                    codec_type = self._get_codec_type(codec)
                    line_style = self._get_codec_line_style(codec_type)
                    trace_name = self._get_trace_name(codec, model)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Convert to kbps
                            y=device_data['vmaf_mean'].tolist(),
                            mode='markers+lines',
                            name=trace_name,
                            line=dict(color=color, dash=line_style, width=2),
                            marker=dict(size=6),
                            legendgroup=trace_name,
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
        # PSNR plot - use calculated_bitrate_bps for X-axis
        if 'psnr' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            # Filter out invalid PSNR values
            valid_psnr_data = combined_df[combined_df['psnr'] != -1]
            if not valid_psnr_data.empty:
                for codec in valid_psnr_data['codec'].unique():
                    codec_data = valid_psnr_data[valid_psnr_data['codec'] == codec].sort_values('calculated_bitrate_bps')
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device]
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                        
                        # Get consistent color and line style
                        color = self._get_device_color(model)
                        codec_type = self._get_codec_type(codec)
                        line_style = self._get_codec_line_style(codec_type)
                        trace_name = self._get_trace_name(codec, model)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Convert to kbps
                                y=device_data['psnr'].tolist(),
                                mode='markers+lines',
                                name=trace_name,
                                line=dict(color=color, dash=line_style, width=2),
                                marker=dict(size=6),
                                legendgroup=trace_name,
                                showlegend=False
                            ),
                            row=1, col=2
                        )
        
        # SSIM plot - use calculated_bitrate_bps for X-axis
        if 'ssim' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            # Filter out invalid SSIM values
            valid_ssim_data = combined_df[combined_df['ssim'] != -1]
            if not valid_ssim_data.empty:
                for codec in valid_ssim_data['codec'].unique():
                    codec_data = valid_ssim_data[valid_ssim_data['codec'] == codec].sort_values('calculated_bitrate_bps')
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device]
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                        
                        # Get consistent color and line style
                        color = self._get_device_color(model)
                        codec_type = self._get_codec_type(codec)
                        line_style = self._get_codec_line_style(codec_type)
                        trace_name = self._get_trace_name(codec, model)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Convert to kbps
                                y=device_data['ssim'].tolist(),
                                mode='markers+lines',
                                name=trace_name,
                                line=dict(color=color, dash=line_style, width=2),
                                marker=dict(size=6),
                                legendgroup=trace_name,
                                showlegend=False
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
                    color = self._get_device_color(model)
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
        
        # Bitrate Accuracy plot (as line graph)
        if 'bitrate_bps' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device].sort_values('bitrate_bps')
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Get consistent color and line style
                    color = self._get_device_color(model)
                    codec_type = self._get_codec_type(codec)
                    line_style = self._get_codec_line_style(codec_type)
                    trace_name = self._get_trace_name(codec, model)
                    
                    # Calculate accuracy percentage
                    target = device_data['bitrate_bps']
                    actual = device_data['calculated_bitrate_bps']
                    accuracy = ((actual - target) / target * 100).tolist()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=(target / 1000).tolist(),  # Target bitrate
                            y=accuracy,
                            mode='markers+lines',
                            name=trace_name,
                            line=dict(color=color, dash=line_style, width=2),
                            marker=dict(size=6),
                            legendgroup=trace_name,
                            showlegend=False
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
        fig.update_xaxes(title_text="Actual Bitrate (kbps)", row=1, col=1)
        fig.update_yaxes(title_text="VMAF", row=1, col=1)
        fig.update_xaxes(title_text="Actual Bitrate (kbps)", row=1, col=2)
        fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text="Actual Bitrate (kbps)", row=2, col=1)
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
            '39251FDJH0093R': '#8c564b'  # Brown
        }
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

    def create_performance_plots(self, test_results: List[TestResult]) -> go.Figure:
        """Create performance plots from encoder statistics CSV files"""
        
        # Create subplots for performance metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Encoding Latency (Aggregated)", "Processing Framerate (Aggregated)", 
                          "Pipeline Depth Over Time", "Bitrate Variability"),
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
                            df = pd.read_csv(csv_file)
                            if not df.empty:
                                # Add device info to the dataframe
                                df['device_serial'] = result.device_serial
                                df['test_name'] = result.test_name
                                all_data.append(df)
        
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
        if 'proctime' in combined_df.columns and 'rel_pts' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Convert nanoseconds to milliseconds
                    latency_ms = (device_data['proctime'] / 1_000_000).values
                    time_sec = (device_data['rel_pts'] / 1000).values
                    
                    # Sort by time for proper aggregation
                    sort_idx = np.argsort(time_sec)
                    time_sec = time_sec[sort_idx]
                    latency_ms = latency_ms[sort_idx]
                    
                    # Calculate statistics
                    mean_latency = np.mean(latency_ms)
                    p50 = np.percentile(latency_ms, 50)
                    p95 = np.percentile(latency_ms, 95)
                    p99 = np.percentile(latency_ms, 99)
                    
                    # Create rolling statistics for smooth curves
                    window_size = min(50, len(latency_ms) // 10) if len(latency_ms) > 10 else 5
                    rolling_mean = pd.Series(latency_ms).rolling(window=window_size, center=True).mean()
                    rolling_std = pd.Series(latency_ms).rolling(window=window_size, center=True).std()
                    
                    # Remove NaN values
                    valid_idx = ~np.isnan(rolling_mean)
                    time_valid = time_sec[valid_idx]
                    mean_valid = rolling_mean[valid_idx]
                    std_valid = rolling_std[valid_idx]
                    
                    # Create consistent trace naming for filtering
                    trace_name = self._get_trace_name(codec, model)
                    codec_type = self._get_codec_type(codec)
                    
                    # Get consistent color and line style
                    color = self._get_device_color(model)
                    line_style = self._get_codec_line_style(codec_type)
                    
                    # Add main line
                    fig.add_trace(
                        go.Scatter(
                            x=time_valid.tolist(),
                            y=mean_valid.tolist(),
                            mode='lines',
                            name=trace_name,
                            line=dict(width=2, color=color, dash=line_style),
                            legendgroup=trace_name,
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
                    # Add confidence interval (upper bound)
                    fig.add_trace(
                        go.Scatter(
                            x=time_valid.tolist(),
                            y=(mean_valid + std_valid).tolist(),
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
                            x=time_valid.tolist(),
                            y=(mean_valid - std_valid).tolist(),
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
        if 'proc_fps' in combined_df.columns and 'rel_pts' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    fps = device_data['proc_fps'].values
                    time_sec = (device_data['rel_pts'] / 1000).values
                    
                    # Sort by time for proper aggregation
                    sort_idx = np.argsort(time_sec)
                    time_sec = time_sec[sort_idx]
                    fps = fps[sort_idx]
                    
                    # Calculate statistics
                    mean_fps = np.mean(fps)
                    p50 = np.percentile(fps, 50)
                    p95 = np.percentile(fps, 95)
                    p99 = np.percentile(fps, 99)
                    
                    # Create rolling statistics for smooth curves
                    window_size = min(50, len(fps) // 10) if len(fps) > 10 else 5
                    rolling_mean = pd.Series(fps).rolling(window=window_size, center=True).mean()
                    rolling_std = pd.Series(fps).rolling(window=window_size, center=True).std()
                    
                    # Remove NaN values
                    valid_idx = ~np.isnan(rolling_mean)
                    time_valid = time_sec[valid_idx]
                    mean_valid = rolling_mean[valid_idx]
                    std_valid = rolling_std[valid_idx]
                    
                    # Create consistent trace naming for filtering
                    trace_name = self._get_trace_name(codec, model)
                    codec_type = self._get_codec_type(codec)
                    
                    # Get consistent color and line style
                    color = self._get_device_color(model)
                    line_style = self._get_codec_line_style(codec_type)
                    
                    # Add main line
                    fig.add_trace(
                        go.Scatter(
                            x=time_valid.tolist(),
                            y=mean_valid.tolist(),
                            mode='lines',
                            name=trace_name,
                            line=dict(width=2, color=color, dash=line_style),
                            legendgroup=trace_name,
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                    
                    # Add confidence interval (upper bound)
                    fig.add_trace(
                        go.Scatter(
                            x=time_valid.tolist(),
                            y=(mean_valid + std_valid).tolist(),
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
                            x=time_valid.tolist(),
                            y=(mean_valid - std_valid).tolist(),
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
                    color = self._get_device_color(model)
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
        
        # Plot 4: Bitrate Variability
        if 'bitrate_per_frame_bps' in combined_df.columns and 'rel_pts' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    bitrate_kbps = (device_data['bitrate_per_frame_bps'] / 1000).values
                    time_sec = (device_data['rel_pts'] / 1000).values
                    
                    # Sort by time for proper aggregation
                    sort_idx = np.argsort(time_sec)
                    time_sec = time_sec[sort_idx]
                    bitrate_kbps = bitrate_kbps[sort_idx]
                    
                    # Create consistent trace name for filtering
                    trace_name = self._get_trace_name(codec, model)
                    codec_type = self._get_codec_type(codec)
                    
                    # Get consistent color and line style
                    color = self._get_device_color(model)
                    line_style = self._get_codec_line_style(codec_type)
        
                    fig.add_trace(
                        go.Scatter(
                            x=time_sec.tolist(),
                            y=bitrate_kbps.tolist(),
                            mode='lines',
                            name=trace_name,
                            line=dict(color=color, dash=line_style, width=2),
                            legendgroup=trace_name,
                            showlegend=False
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
    
    def create_comparison_plots(self, test_results: List[TestResult], reference_device: str = None) -> go.Figure:
        """Create comparison plots with delta analysis"""
        
        # Create subplots for comparison
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("VMAF Delta Comparison", "SI/TI Complexity Analysis"),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Read all quality CSV files directly
        all_data = []
        for result in test_results:
            if result.success and hasattr(result, 'test_data') and 'quality_csv' in result.test_data:
                csv_file = result.test_data['quality_csv']
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        # Add device info to the dataframe
                        df['device_serial'] = result.device_serial
                        df['test_name'] = result.test_name
                        all_data.append(df)
        
        if not all_data:
            return fig
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # VMAF Delta Comparison
        if 'vmaf_mean' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            # Choose reference based on selection or first available
            reference_data = None
            if reference_device:
                # Find reference data for selected device
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec]
                    device_data = codec_data[codec_data['device_serial'] == reference_device].sort_values('calculated_bitrate_bps')
                    if len(device_data) >= 2:
                        reference_data = device_data
                        break
            
            # Fallback to first available if no reference selected or found
            if reference_data is None:
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device].sort_values('calculated_bitrate_bps')
                        if len(device_data) >= 2:
                            reference_data = device_data
                            break
                    if reference_data is not None:
                        break
            
            if reference_data is not None:
                # Interpolate reference curve
                ref_bitrates = reference_data['calculated_bitrate_bps'].values
                ref_vmaf = reference_data['vmaf_mean'].values
                
                # Create interpolation function
                ref_interp = np.interp
                
                # Compare all other combinations
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device].sort_values('calculated_bitrate_bps')
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                        
                        if len(device_data) >= 2:
                            # Calculate delta
                            bitrates = device_data['calculated_bitrate_bps'].values
                            vmaf_values = device_data['vmaf_mean'].values
                            
                            # Interpolate reference VMAF at same bitrates
                            ref_vmaf_interp = np.interp(bitrates, ref_bitrates, ref_vmaf)
                            delta_vmaf = vmaf_values - ref_vmaf_interp
        
                            # Get consistent color and line style
                            color = self._get_device_color(model)
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
                                ),
                                row=1, col=1
                            )
        
        # SI/TI Complexity Analysis
        if 'si_avg' in combined_df.columns and 'ti_avg' in combined_df.columns:
            # Filter out invalid values
            valid_data = combined_df[(combined_df['si_avg'] != -1) & (combined_df['ti_avg'] != -1)]
            
            if not valid_data.empty:
                for codec in valid_data['codec'].unique():
                    codec_data = valid_data[valid_data['codec'] == codec]
                    for device in codec_data['device_serial'].unique():
                        device_data = codec_data[codec_data['device_serial'] == device]
                        model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
        
                        # Get consistent color and line style
                        color = self._get_device_color(model)
                        codec_type = self._get_codec_type(codec)
                        trace_name = self._get_trace_name(codec, model)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=device_data['ti_avg'].tolist(),
                                y=device_data['si_avg'].tolist(),
                                mode='markers',
                                name=trace_name,
                                marker=dict(size=8, color=color),
                                hovertemplate=f'<b>{codec} ({model})</b><br>' +
                                            'TI: %{x}<br>' +
                                            'SI: %{y}<br>' +
                                            '<extra></extra>',
                                legendgroup=trace_name,
                                showlegend=False
                            ),
                            row=2, col=1
                        )
        
        # Update layout
        fig.update_layout(
            title="Comparison Analysis",
            height=800,
            showlegend=True,
            autosize=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Actual Bitrate (kbps)", row=1, col=1)
        fig.update_yaxes(title_text="VMAF Delta", row=1, col=1)
        fig.update_xaxes(title_text="TI (Temporal Information)", row=2, col=1)
        fig.update_yaxes(title_text="SI (Spatial Information)", row=2, col=1)
        
        return fig
    
    def generate_interactive_report(self, test_results: List[TestResult], 
                                  stats_files: List[str] = None) -> str:
        """Generate an interactive HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ava_report_{timestamp}.html"
        
        # Create all plots
        quality_fig = self.create_quality_plots_from_csv(test_results)
        performance_fig = self.create_performance_plots(test_results)
        comparison_fig = self.create_comparison_plots(test_results)
        
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
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'Quality')">Quality Metrics</button>
                <button class="tablinks" onclick="openTab(event, 'Performance')">Performance</button>
                <button class="tablinks" onclick="openTab(event, 'Comparison')">Comparison</button>
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
                            <h4>Actions:</h4>
                            <button onclick="showAllComparisonTraces()">Show All</button>
                            <button onclick="hideAllComparisonTraces()">Hide All</button>
                        </div>
                    </div>
                </div>
                <div id="comparison-plots"></div>
            </div>
            
            <script>
                // Global variables for plot data
                var qualityData = {quality_fig.to_json()};
                var performanceData = {performance_fig.to_json()};
                var comparisonData = {comparison_fig.to_json()};
                
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
                }}
                
                // Extract unique devices from quality data
                function extractDevices() {{
                    var devices = new Set();
                    qualityData.data.forEach(function(trace) {{
                        var name = trace.name;
                        // Only process traces that have a name and match the expected format
                        if (name && typeof name === 'string') {{
                            // Extract device from trace name (format: "codec (device)")
                            var match = name.match(/\\(([^)]+)\\)/);
                            if (match) {{
                                devices.add(match[1]);
                            }}
                        }}
                    }});
                    return Array.from(devices).sort();
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
                    console.log('Performance data:', performanceData);
                    performanceData.data.forEach(function(trace) {{
                        try {{
                            var name = trace.name;
                            console.log('Performance trace name:', name);
                            // Only process traces that have a name and match the expected format
                            if (name && typeof name === 'string' && name.length > 0) {{
                                // Extract device from trace name (format: "codec (device)")
                                var match = name.match(/\\(([^)]+)\\)/);
                                if (match) {{
                                    devices.add(match[1]);
                                    console.log('Found device:', match[1]);
                                }}
                            }} else {{
                                console.log('Skipping trace with invalid name:', name);
                            }}
                        }} catch (error) {{
                            console.log('Error processing trace:', error, trace);
                        }}
                    }});
                    console.log('Extracted devices:', Array.from(devices));
                    return Array.from(devices).sort();
                }}
                
                // Create performance device filter checkboxes
                function createPerformanceDeviceFilters() {{
                    var devices = extractPerformanceDevices();
                    var container = document.getElementById('perf-device-filters');
                    console.log('Creating performance device filters for devices:', devices);
                    console.log('Container element:', container);
                    if (devices.length === 0) {{
                        console.log('No devices found for performance filters');
                        container.innerHTML = '<p>No device data available</p>';
                        return;
                    }}
                    devices.forEach(function(device) {{
                        var label = document.createElement('label');
                        label.innerHTML = '<input type="checkbox" id="perf-filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_') + '" checked onchange="filterPerformanceTraces()"> ' + device;
                        container.appendChild(label);
                        container.appendChild(document.createElement('br'));
                    }});
                }}
                
                // Extract unique devices from comparison data
                function extractComparisonDevices() {{
                    var devices = new Set();
                    console.log('Comparison data:', comparisonData);
                    comparisonData.data.forEach(function(trace) {{
                        var name = trace.name;
                        console.log('Comparison trace name:', name);
                        // Only process traces that have a name and match the expected format
                        if (name && typeof name === 'string') {{
                            // Extract device from trace name (format: "codec (device)")
                            var match = name.match(/\\(([^)]+)\\)/);
                            if (match) {{
                                devices.add(match[1]);
                                console.log('Found comparison device:', match[1]);
                            }}
                        }}
                    }});
                    console.log('Extracted comparison devices:', Array.from(devices));
                    return Array.from(devices).sort();
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
                    var devices = extractComparisonDevices();
                    var selector = document.getElementById('reference-selector');
                    console.log('Creating reference selector for devices:', devices);
                    console.log('Reference selector element:', selector);
                    
                    if (!selector) {{
                        console.log('Reference selector element not found');
                        return;
                    }}
                    
                    // If no comparison devices, try to get devices from quality data
                    if (devices.length === 0) {{
                        console.log('No comparison devices found, trying quality data');
                        devices = extractDevices();
                        console.log('Quality devices for reference:', devices);
                    }}
                    
                    if (devices.length === 0) {{
                        console.log('No devices found for reference selector');
                        selector.innerHTML = '<option value="">No devices available</option>';
                        return;
                    }}
                    
                    // Clear existing options except the first one
                    while (selector.children.length > 1) {{
                        selector.removeChild(selector.lastChild);
                    }}
                    
                    devices.forEach(function(device) {{
                        var option = document.createElement('option');
                        option.value = device;
                        option.textContent = device;
                        selector.appendChild(option);
                    }});
                    
                    console.log('Reference selector populated with', devices.length, 'devices');
                }}
                
                // Get codec type from trace name
                function getCodecType(traceName) {{
                    var name = traceName.toLowerCase();
                    if (name.includes('av1')) return 'av1';
                    if (name.includes('hevc') || name.includes('h265')) return 'hevc';
                    if (name.includes('avc') || name.includes('h264')) return 'avc';
                    if (name.includes('vp8')) return 'vp8';
                    return 'other';
                }}
                
                // Get device from trace name
                function getDevice(traceName) {{
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
                    var visibleTraces = [];
                    var codecFilters = {{
                        av1: document.getElementById('perf-filter-av1').checked,
                        hevc: document.getElementById('perf-filter-hevc').checked,
                        avc: document.getElementById('perf-filter-avc').checked,
                        vp8: document.getElementById('perf-filter-vp8').checked
                    }};
                    
                    // Check if any codec filters are enabled
                    var anyCodecEnabled = Object.values(codecFilters).some(function(enabled) {{ return enabled; }});
                    
                    performanceData.data.forEach(function(trace, index) {{
                        var codecType = getCodecType(trace.name);
                        var device = getDevice(trace.name);
                        var deviceFilterId = 'perf-filter-device-' + device.replace(/[^a-zA-Z0-9]/g, '_');
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
                        Plotly.restyle('performance-plots', {{visible: 'legendonly'}}, {{}});
                    }} else {{
                        Plotly.restyle('performance-plots', {{visible: 'legendonly'}}, {{}});
                        Plotly.restyle('performance-plots', {{visible: true}}, visibleTraces);
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
                    if (selectedReference) {{
                        console.log('Reference selected:', selectedReference);
                        // Regenerate comparison plots with new reference
                        regenerateComparisonPlots(selectedReference);
                    }}
                }}
                
                // Store raw VMAF data for delta calculations
                var rawVmafData = {{}};
                
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
                
                // Store original SI/TI data
                var originalSiTiData = [];
                
                // Extract SI/TI data from comparison plots
                function extractSiTiData() {{
                    originalSiTiData = [];
                    if (comparisonData && comparisonData.data) {{
                        console.log('Looking for SI/TI data in', comparisonData.data.length, 'traces');
                        comparisonData.data.forEach(function(trace, index) {{
                            console.log('Trace', index, ':', trace.name, 'xaxis:', trace.xaxis, 'yaxis:', trace.yaxis);
                            // Look for SI/TI traces (they should be in the second subplot)
                            if (trace.name && trace.x && trace.y && trace.x.length > 0 && trace.y.length > 0) {{
                                // Check if this looks like SI/TI data (scatter plot with reasonable ranges)
                                var xValues = trace.x;
                                var yValues = trace.y;
                                var isSiTi = xValues.every(function(x) {{ return x >= 0 && x <= 100; }}) && 
                                            yValues.every(function(y) {{ return y >= 0 && y <= 100; }});
                                
                                console.log('Trace', index, 'SI/TI check:', isSiTi, 'x range:', Math.min(...xValues), '-', Math.max(...xValues), 'y range:', Math.min(...yValues), '-', Math.max(...yValues));
                                
                                if (isSiTi) {{
                                    originalSiTiData.push({{
                                        x: trace.x,
                                        y: trace.y,
                                        name: trace.name,
                                        mode: trace.mode || 'markers',
                                        marker: trace.marker || {{ size: 8 }},
                                        hovertemplate: trace.hovertemplate,
                                        xaxis: 'x2',
                                        yaxis: 'y2'
                                    }});
                                }}
                            }}
                        }});
                    }}
                    console.log('Extracted SI/TI data:', originalSiTiData);
                }}
                
                // Store original VMAF delta data for reference switching
                var originalVmafDeltaData = [];
                
                // Extract original VMAF delta data from comparison plots
                function extractOriginalVmafDeltaData() {{
                    originalVmafDeltaData = [];
                    if (comparisonData && comparisonData.data) {{
                        console.log('Extracting original VMAF delta data from', comparisonData.data.length, 'traces');
                        comparisonData.data.forEach(function(trace, index) {{
                            // Look for VMAF delta traces (they should be in the first subplot)
                            if (trace.xaxis === 'x' && trace.yaxis === 'y' && trace.name && trace.x && trace.y) {{
                                console.log('VMAF delta trace', index, ':', trace.name, 'y range:', Math.min(...trace.y), '-', Math.max(...trace.y));
                                originalVmafDeltaData.push({{
                                    x: trace.x,
                                    y: trace.y,
                                    name: trace.name,
                                    mode: trace.mode || 'markers+lines',
                                    line: trace.line || {{ width: 2 }},
                                    marker: trace.marker || {{ size: 6 }},
                                    xaxis: 'x',
                                    yaxis: 'y'
                                }});
                            }}
                        }});
                    }}
                    console.log('Extracted original VMAF delta data:', originalVmafDeltaData);
                }}
                
                // Regenerate comparison plots with selected reference
                function regenerateComparisonPlots(referenceDevice) {{
                    console.log('Updating reference to:', referenceDevice);
                    
                    if (!referenceDevice) {{
                        console.log('No reference device selected');
                        return;
                    }}
                    
                    // Find the reference device trace in original data
                    var referenceTrace = null;
                    var referenceDeviceName = '';
                    originalVmafDeltaData.forEach(function(trace) {{
                        if (trace.name && trace.name.includes(referenceDevice)) {{
                            referenceTrace = trace;
                            referenceDeviceName = trace.name;
                        }}
                    }});
                    
                    if (!referenceTrace) {{
                        console.log('Reference device not found in original data:', referenceDevice);
                        return;
                    }}
                    
                    console.log('Found reference trace:', referenceDeviceName, 'with', referenceTrace.y.length, 'data points');
                    
                    // Create new traces with adjusted deltas
                    var newTraces = [];
                    originalVmafDeltaData.forEach(function(trace) {{
                        var adjustedY = [];
                        
                        if (trace.name === referenceDeviceName) {{
                            // Reference device should show as 0 delta
                            adjustedY = new Array(trace.y.length).fill(0);
                        }} else {{
                            // Other devices: subtract reference deltas from their deltas
                            for (var i = 0; i < trace.y.length; i++) {{
                                var refY = referenceTrace.y[i] || 0;
                                var deviceY = trace.y[i] || 0;
                                var adjustedDelta = deviceY - refY;
                                adjustedY.push(adjustedDelta);
                            }}
                        }}
                        
                        newTraces.push({{
                            x: trace.x,
                            y: adjustedY,
                            mode: trace.mode,
                            name: trace.name,
                            type: 'scatter',
                            line: trace.line,
                            marker: trace.marker,
                            xaxis: 'x',
                            yaxis: 'y'
                        }});
                    }});
                    
                    // Add SI/TI data to the second subplot
                    originalSiTiData.forEach(function(siTiTrace) {{
                        newTraces.push({{
                            x: siTiTrace.x,
                            y: siTiTrace.y,
                            mode: siTiTrace.mode,
                            name: siTiTrace.name,
                            type: 'scatter',
                            marker: siTiTrace.marker,
                            hovertemplate: siTiTrace.hovertemplate,
                            xaxis: 'x2',
                            yaxis: 'y2'
                        }});
                    }});
                    
                    // Update the plot with both VMAF deltas and SI/TI
                    Plotly.react('comparison-plots', newTraces, comparisonData.layout, {{responsive: true}});
                    
                    // Update the plot title
                    var update = {{
                        'title': 'VMAF Delta Comparison (Reference: ' + referenceDevice + ')'
                    }};
                    Plotly.relayout('comparison-plots', update);
                    
                    console.log('✅ REFERENCE SWITCHING ACTIVE: Updated comparison plot with', newTraces.length, 'traces');
                    console.log('Reference device:', referenceDeviceName, 'now shows as 0 delta');
                }}
                
                // Initialize plots
                function initializePlots() {{
                    // Load quality plots
                    Plotly.newPlot('quality-plots', qualityData.data, qualityData.layout, {{responsive: true}});
                    
                    // Load performance plots
                    Plotly.newPlot('performance-plots', performanceData.data, performanceData.layout, {{responsive: true}});
                    
                    // Load comparison plots
                    Plotly.newPlot('comparison-plots', comparisonData.data, comparisonData.layout, {{responsive: true}});
                    
                    // Extract raw VMAF data for delta calculations
                    extractRawVmafData();
                    
                    // Extract SI/TI data for complexity analysis
                    extractSiTiData();
                    
                    // Extract original VMAF delta data for reference switching
                    extractOriginalVmafDeltaData();
                    
                    // Create device filters after plots are loaded
                    createDeviceFilters();
                    createPerformanceDeviceFilters();
                    createComparisonDeviceFilters();
                    createReferenceSelector();
                }}
                
                // Initialize when page loads
                window.onload = function() {{
                    initializePlots();
                }};
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Interactive report saved to: {output_file}")
        return str(output_file)

    def generate_comprehensive_report(self, test_results: List[TestResult], 
                                    stats_files: List[str] = None) -> Dict[str, str]:
        """Generate comprehensive reports - wrapper for interactive report"""
        interactive_report = self.generate_interactive_report(test_results, stats_files)
        
        return {
            "interactive": interactive_report,
            "static": interactive_report,  # Same as interactive for now
            "json": ""  # Not implemented in minimal version
        }
