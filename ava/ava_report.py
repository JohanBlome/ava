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
    quality_metrics: List[Dict[str, Any]] = None
    qp_values: Dict[str, Any] = None
    test_data: Dict[str, Any] = None

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
                    
                    fig.add_trace(
                        go.Scatter(
                            x=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Convert to kbps
                            y=device_data['vmaf_mean'].tolist(),
                            mode='markers+lines',
                            name=f'{codec} ({model})',
                            line=dict(dash='solid')
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
                        
                        fig.add_trace(
                            go.Scatter(
                                x=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Convert to kbps
                                y=device_data['psnr'].tolist(),
                                mode='markers+lines',
                                name=f'PSNR - {codec} ({model})',
                                line=dict(dash='dash')
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
                        
                        fig.add_trace(
                            go.Scatter(
                                x=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Convert to kbps
                                y=device_data['ssim'].tolist(),
                                mode='markers+lines',
                                name=f'SSIM - {codec} ({model})',
                                line=dict(dash='dot')
                            ),
                            row=2, col=1
                        )
        
        # Target vs Actual Bitrate plot
        if 'bitrate_bps' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    fig.add_trace(
                        go.Scatter(
                            x=(device_data['bitrate_bps'] / 1000).tolist(),  # Target bitrate
                            y=(device_data['calculated_bitrate_bps'] / 1000).tolist(),  # Actual bitrate
                            mode='markers',
                            name=f'Bitrate - {codec} ({model})',
                            marker=dict(size=8)
                        ),
                        row=3, col=1
                    )
        
        # Bitrate Accuracy plot
        if 'bitrate_bps' in combined_df.columns and 'calculated_bitrate_bps' in combined_df.columns:
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                for device in codec_data['device_serial'].unique():
                    device_data = codec_data[codec_data['device_serial'] == device]
                    model = device_data['model'].iloc[0] if 'model' in device_data.columns else device
                    
                    # Calculate accuracy percentage
                    target = device_data['bitrate_bps']
                    actual = device_data['calculated_bitrate_bps']
                    accuracy = ((actual - target) / target * 100).tolist()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=(target / 1000).tolist(),  # Target bitrate
                            y=accuracy,
                            mode='markers',
                            name=f'Accuracy - {codec} ({model})',
                            marker=dict(size=8)
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

    def generate_interactive_report(self, test_results: List[TestResult], 
                                  stats_files: List[str] = None) -> str:
        """Generate an interactive HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ava_report_{timestamp}.html"
        
        # Create quality plots
        quality_fig = self.create_quality_plots_from_csv(test_results)
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AVA Quality Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>AVA Quality Metrics Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div id="quality-plots"></div>
            
            <script>
                var plotData = {quality_fig.to_json()};
                Plotly.newPlot('quality-plots', plotData.data, plotData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Interactive report saved to: {output_file}")
        return str(output_file)
