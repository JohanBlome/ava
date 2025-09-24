#!/usr/bin/env python3

"""
AVA Reporting System

This module provides interactive plotting and static reporting capabilities
for video codec test results using Plotly and other visualization tools.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available. Install with: pip install pandas")


@dataclass
class TestResult:
    """Container for test result data"""
    test_name: str
    device_serial: str
    encoder: str
    success: bool
    duration: float
    quality_metrics: Optional[Dict[str, Any]] = None
    test_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    output_files: Optional[List[str]] = None
    bitrate: Optional[float] = None
    qp_values: Optional[Dict[str, Any]] = None
    source_file: Optional[str] = None
    resolution: Optional[str] = None
    framerate: Optional[float] = None


class ReportGenerator:
    """Generates interactive and static reports for test results"""
    
    def __init__(self, output_dir: str = "reports", debug: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debug information"""
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
    
    def analyze_quality_metrics(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze quality metrics from test results"""
        quality_data = {
            "vmaf_data": [],
            "psnr_data": [],
            "ssim_data": [],
            "qp_data": [],
            "bitrate_data": [],
            "encoder_comparison": {},
            "device_comparison": {},
            "source_comparison": {}
        }
        
        for result in test_results:
            if not result.success or not result.quality_metrics:
                continue
                
            # Extract quality metrics - quality_metrics is now a list of dictionaries
            for metrics in result.quality_metrics:
                if "vmaf" in metrics:
                quality_data["vmaf_data"].append({
                        "encoder": metrics.get("codec", result.encoder),  # Use codec from CSV, fallback to result.encoder
                    "device": result.device_serial,
                        "model": metrics.get("model", result.device_serial),  # Use model from CSV if available
                        "bitrate": metrics.get("calculated_bitrate_bps") or metrics.get("bitrate_bps"),
                        "target_bitrate": metrics.get("bitrate_bps"),
                        "actual_bitrate": metrics.get("calculated_bitrate_bps"),
                        "source": metrics.get("media"),
                        "resolution": metrics.get("resolution"),
                        "framerate": metrics.get("framerate_fps"),
                    "vmaf": metrics["vmaf"],
                        "si_avg": metrics.get("si_avg"),  # Add SI/TI data for complexity analysis
                        "ti_avg": metrics.get("ti_avg"),
                    "test_name": result.test_name
                })
            
                if "psnr" in metrics and metrics["psnr"] is not None and metrics["psnr"] != -1:
                quality_data["psnr_data"].append({
                        "encoder": metrics.get("codec", result.encoder),  # Use codec from CSV, fallback to result.encoder
                    "device": result.device_serial,
                        "model": metrics.get("model", result.device_serial),  # Use model from CSV if available
                        "bitrate": metrics.get("calculated_bitrate_bps") or metrics.get("bitrate_bps"),
                        "target_bitrate": metrics.get("bitrate_bps"),
                        "actual_bitrate": metrics.get("calculated_bitrate_bps"),
                        "source": metrics.get("media"),
                        "resolution": metrics.get("resolution"),
                        "framerate": metrics.get("framerate_fps"),
                    "psnr": metrics["psnr"],
                    "test_name": result.test_name
                })
            
                if "ssim" in metrics and metrics["ssim"] is not None and metrics["ssim"] != -1:
                quality_data["ssim_data"].append({
                        "encoder": metrics.get("codec", result.encoder),  # Use codec from CSV, fallback to result.encoder
                    "device": result.device_serial,
                        "model": metrics.get("model", result.device_serial),  # Use model from CSV if available
                        "bitrate": metrics.get("calculated_bitrate_bps") or metrics.get("bitrate_bps"),
                        "target_bitrate": metrics.get("bitrate_bps"),
                        "actual_bitrate": metrics.get("calculated_bitrate_bps"),
                        "source": metrics.get("media"),
                        "resolution": metrics.get("resolution"),
                        "framerate": metrics.get("framerate_fps"),
                    "ssim": metrics["ssim"],
                    "test_name": result.test_name
                })
            
                # Add bitrate comparison data - be more flexible with data availability
                target_bitrate = metrics.get("bitrate_bps")
                actual_bitrate = metrics.get("calculated_bitrate_bps")
                
                # Add bitrate data if we have at least one bitrate value
                if target_bitrate or actual_bitrate:
                    # Use actual_bitrate as target if target_bitrate is missing
                    if not target_bitrate and actual_bitrate:
                        target_bitrate = actual_bitrate
                    # Use target_bitrate as actual if actual_bitrate is missing  
                    if not actual_bitrate and target_bitrate:
                        actual_bitrate = target_bitrate
                        
                    quality_data["bitrate_data"].append({
                        "encoder": metrics.get("codec", result.encoder),
                        "device": result.device_serial,
                        "model": metrics.get("model", result.device_serial),  # Use model from CSV if available
                        "target_bitrate": target_bitrate,
                        "actual_bitrate": actual_bitrate,
                        "source": metrics.get("media"),
                        "resolution": metrics.get("resolution"),
                        "framerate": metrics.get("framerate_fps"),
                        "test_name": result.test_name
                    })
        
        # Handle QP values (outside the metrics loop)
        for result in test_results:
            if result.success and result.qp_values:
                quality_data["qp_data"].append({
                    "encoder": result.encoder,
                    "device": result.device_serial,
                    "bitrate": result.bitrate,
                    "source": result.source_file,
                    "qp_i": result.qp_values.get("i", {}),
                    "qp_p": result.qp_values.get("p", {}),
                    "qp_b": result.qp_values.get("b", {}),
                    "test_name": result.test_name
                })
        
        return quality_data
    
    def create_quality_plots_from_csv(self, test_results: List[Any]) -> go.Figure:
        """Create quality plots directly from CSV files - no redundant data structures"""
        if not PLOTLY_AVAILABLE:
            return None
            
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

    def create_quality_plots(self, quality_data: Dict[str, Any]) -> go.Figure:
        """Create interactive quality metrics plots"""
        if not PLOTLY_AVAILABLE:
            return None
            
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
        
        # VMAF plot
        if quality_data["vmaf_data"]:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
            for encoder in vmaf_df["encoder"].unique():
                encoder_data = vmaf_df[vmaf_df["encoder"] == encoder]
                for device in encoder_data["device"].unique():
                    device_data = encoder_data[encoder_data["device"] == device].sort_values("bitrate")  # Sort by bitrate
                    # Convert to lists to ensure proper serialization
                    x_values = (device_data["bitrate"] / 1000).tolist()  # Convert bps to kbps
                    y_values = device_data["vmaf"].tolist()
                    
                    # Use model name if available, fallback to device serial
                    model = device_data["model"].iloc[0] if "model" in device_data.columns else device
                    
                    # For scaling with many codecs/devices, use shorter names
                    unique_encoders = vmaf_df["encoder"].nunique()
                    unique_devices = vmaf_df["device"].nunique()
                    
                    if unique_encoders * unique_devices > 12:  # Too many combinations
                        # Use shorter naming to avoid legend clutter
                        display_name = f'{encoder.split(".")[-1]} ({model[:8]})'  # Short encoder + truncated model
                    else:
                        display_name = f'VMAF - {encoder} ({model})'
                    
                    trace = go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='markers+lines',
                        name=display_name,
                        line=dict(dash='solid')
                )
                    fig.add_trace(trace, row=1, col=1)
        
        # PSNR plot
        if quality_data["psnr_data"]:
            psnr_df = pd.DataFrame(quality_data["psnr_data"])
            for encoder in psnr_df["encoder"].unique():
                encoder_data = psnr_df[psnr_df["encoder"] == encoder]
                for device in encoder_data["device"].unique():
                    device_data = encoder_data[encoder_data["device"] == device].sort_values("bitrate")  # Sort by bitrate
                    x_values = (device_data["bitrate"] / 1000).tolist()  # Convert bps to kbps
                    y_values = device_data["psnr"].tolist()
                    # Use model name if available, fallback to device serial
                    model = device_data["model"].iloc[0] if "model" in device_data.columns else device
                fig.add_trace(
                    go.Scatter(
                            x=x_values,
                            y=y_values,
                        mode='markers+lines',
                            name=f'PSNR - {encoder} ({model})',
                        line=dict(dash='dash')
                    ),
                    row=1, col=2
                )
        
        # SSIM plot
        if quality_data["ssim_data"]:
            ssim_df = pd.DataFrame(quality_data["ssim_data"])
            for encoder in ssim_df["encoder"].unique():
                encoder_data = ssim_df[ssim_df["encoder"] == encoder]
                for device in encoder_data["device"].unique():
                    device_data = encoder_data[encoder_data["device"] == device].sort_values("bitrate")  # Sort by bitrate
                    x_values = (device_data["bitrate"] / 1000).tolist()  # Convert bps to kbps
                    y_values = device_data["ssim"].tolist()
                    # Use model name if available, fallback to device serial
                    model = device_data["model"].iloc[0] if "model" in device_data.columns else device
                fig.add_trace(
                    go.Scatter(
                            x=x_values,
                            y=y_values,
                        mode='markers+lines',
                            name=f'SSIM - {encoder} ({model})',
                        line=dict(dash='dot')
                    ),
                    row=2, col=1
                )
        
        # QP Statistics plot
        if quality_data["qp_data"]:
            qp_df = pd.DataFrame(quality_data["qp_data"])
            qp_avg = qp_df.groupby("encoder").agg({
                "qp_i": lambda x: x.apply(lambda y: y.get("avg", 0) if isinstance(y, dict) else 0).mean(),
                "qp_p": lambda x: x.apply(lambda y: y.get("avg", 0) if isinstance(y, dict) else 0).mean(),
                "qp_b": lambda x: x.apply(lambda y: y.get("avg", 0) if isinstance(y, dict) else 0).mean()
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=qp_avg["encoder"],
                    y=qp_avg["qp_i"],
                    name="QP I-frames",
                    marker_color='blue'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(
                    x=qp_avg["encoder"],
                    y=qp_avg["qp_p"],
                    name="QP P-frames",
                    marker_color='green'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(
                    x=qp_avg["encoder"],
                    y=qp_avg["qp_b"],
                    name="QP B-frames",
                    marker_color='red'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Video Codec Quality Analysis",
            height=1200,  # Increased height for 3 rows
            showlegend=True,
            # Enable interactive features
            autosize=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Bitrate comparison plots
        if quality_data["bitrate_data"]:
            bitrate_df = pd.DataFrame(quality_data["bitrate_data"])
            
            # Define consistent colors for each encoder-device combination
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color_index = 0
            
            for encoder in bitrate_df["encoder"].unique():
                encoder_data = bitrate_df[bitrate_df["encoder"] == encoder]
                for device in encoder_data["device"].unique():
                    device_data = encoder_data[encoder_data["device"] == device].sort_values("target_bitrate")
                    
                    # Use model name if available, fallback to device serial
                    # If model is the same as device serial, show both for clarity
                    model = device_data["model"].iloc[0] if "model" in device_data.columns else device
                    if model == device or not model or model.strip() == "":
                        display_name = f"{device}"  # Just show serial if no model
                    else:
                        display_name = f"{model}"  # Show model name
                    
                    # Assign consistent color
                    color = colors[color_index % len(colors)]
                    color_index += 1
                    
                    # Target vs Actual Bitrate scatter plot
                    target_values = (device_data["target_bitrate"] / 1000).tolist()  # Convert to kbps
                    actual_values = (device_data["actual_bitrate"] / 1000).tolist()  # Convert to kbps
                    
                    fig.add_trace(
                        go.Scatter(
                            x=target_values,
                            y=actual_values,
                            mode='markers+lines',
                            name=f'Bitrate - {encoder} ({display_name})',
                            line=dict(dash='solid', color=color),
                            marker=dict(size=8, color=color)
                        ),
                        row=3, col=1
                    )
                    
                    # Bitrate accuracy (percentage error)
                    if np is not None:
                        accuracy_values = ((np.array(actual_values) - np.array(target_values)) / np.array(target_values) * 100).tolist()
                    else:
                        # Fallback without numpy
                        accuracy_values = [((actual - target) / target * 100) for actual, target in zip(actual_values, target_values)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=target_values,
                            y=accuracy_values,
                            mode='markers+lines',
                            name=f'Accuracy - {encoder} ({display_name})',
                            line=dict(dash='dash', color=color),
                            marker=dict(size=8, color=color)
                        ),
                        row=3, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Bitrate (kbps)", row=1, col=1)
        fig.update_yaxes(title_text="VMAF", row=1, col=1)
        fig.update_xaxes(title_text="Bitrate (kbps)", row=1, col=2)
        fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text="Bitrate (kbps)", row=2, col=1)
        fig.update_yaxes(title_text="SSIM", row=2, col=1)
        fig.update_xaxes(title_text="Encoder", row=2, col=2)
        fig.update_yaxes(title_text="Average QP", row=2, col=2)
        fig.update_xaxes(title_text="Target Bitrate (kbps)", row=3, col=1)
        fig.update_yaxes(title_text="Actual Bitrate (kbps)", row=3, col=1)
        fig.update_xaxes(title_text="Target Bitrate (kbps)", row=3, col=2)
        fig.update_yaxes(title_text="Bitrate Accuracy (%)", row=3, col=2)
        
        return fig
    
    def generate_interactive_report(self, test_results: List[TestResult], 
                                  quality_results: List[Dict[str, Any]] = None) -> str:
        """Generate comprehensive interactive HTML report using Plotly"""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available, generating basic HTML report")
            return self._generate_basic_html_report(test_results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"ava_report_{timestamp}.html"
        
        # Analyze quality metrics
        quality_data = self.analyze_quality_metrics(test_results)
        
        # Collect encoder statistics files
        stats_files = []
        for result in test_results:
            self.logger.info(f"Checking result {result.test_name}: success={result.success}, test_data keys={list(result.test_data.keys()) if result.test_data else 'None'}")
            if result.success and result.test_data and "stats_csv" in result.test_data:
                stats_csv = result.test_data["stats_csv"]
                self.logger.info(f"Found stats_csv for {result.test_name}: {stats_csv}")
                if isinstance(stats_csv, list):
                    stats_files.extend(stats_csv)
                elif isinstance(stats_csv, str) and stats_csv:
                    stats_files.append(stats_csv)
            else:
                self.logger.warning(f"No stats_csv found for {result.test_name}: success={result.success}, has_test_data={bool(result.test_data)}, stats_csv_key={'stats_csv' in result.test_data if result.test_data else False}")
        
        self.logger.info(f"Total stats files collected: {len(stats_files)}")
        for i, file in enumerate(stats_files):
            exists = os.path.exists(file) if file else False
            self.logger.info(f"Stats file {i+1}: {file} (exists: {exists})")
            if file and exists:
                # Check file size
                try:
                    size = os.path.getsize(file)
                    self.logger.info(f"  File size: {size} bytes")
                except Exception as e:
                    self.logger.error(f"  Error getting file size: {e}")
        
        # Create comprehensive dashboard with multiple tabs
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from plotly.offline import plot
        
        # Create main dashboard
        dashboard_html = self._create_dashboard_html(test_results, quality_data, quality_results, stats_files)
        
        with open(report_file, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Interactive report saved to: {report_file}")
        return str(report_file)
    
    def _create_dashboard_html(self, test_results: List[TestResult], 
                             quality_data: Dict[str, Any], 
                              quality_results: List[Dict[str, Any]] = None,
                              stats_files: List[str] = None) -> str:
        """Create comprehensive dashboard HTML"""
        from plotly.offline import plot
        
        # Generate individual plot figures
        quality_fig = self.create_quality_plots_from_csv(test_results)
        comparison_fig = self.create_comparison_plots(quality_data)
        summary_fig = self._create_summary_plots(test_results)
        error_fig = self._create_error_analysis_plots(test_results)
        
        # Generate encoder statistics plots if stats files are available
        encoder_stats_fig = None
        encoder_stats_html = ""
        if stats_files:
            encoder_stats_fig = self.create_encoder_statistics_plots(stats_files)
            encoder_stats_html = plot(encoder_stats_fig, output_type='div', include_plotlyjs=False) if encoder_stats_fig else ""
        
        # If no encoder statistics available, generate aggregated stats from quality data
        if not encoder_stats_html:
            encoder_stats_html = self._create_aggregated_performance_stats(test_results, quality_data)
        
        # Convert figures to HTML
        quality_html = plot(quality_fig, output_type='div', include_plotlyjs=False) if quality_fig else ""
        comparison_html = plot(comparison_fig, output_type='div', include_plotlyjs=False) if comparison_fig else ""
        summary_html = plot(summary_fig, output_type='div', include_plotlyjs=False) if summary_fig else ""
        error_html = plot(error_fig, output_type='div', include_plotlyjs=False) if error_fig else ""
        
        # Create comprehensive HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AVA Video Codec Test Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
                .summary-stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background: #f0f0f0; padding: 20px; border-radius: 5px; text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #333; }}
                .stat-label {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>AVA Video Codec Test Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-number">{len(test_results)}</div>
                    <div class="stat-label">Total Tests</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sum(1 for r in test_results if r.success)}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(test_results) - sum(1 for r in test_results if r.success)}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sum(1 for r in test_results if r.success) / len(test_results) * 100:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'summary')">Summary</button>
                <button class="tablinks" onclick="openTab(event, 'quality')">Quality Metrics</button>
                <button class="tablinks" onclick="openTab(event, 'comparison')">Comparison</button>
                <button class="tablinks" onclick="openTab(event, 'encoder-stats')">Encoder Performance</button>
                <button class="tablinks" onclick="openTab(event, 'errors')">Error Analysis</button>
            </div>
            
            <div id="summary" class="tabcontent" style="display:block">
                <h2>Test Summary</h2>
                {summary_html}
            </div>
            
            <div id="quality" class="tabcontent">
                <h2>Quality Metrics Analysis</h2>
                {quality_html}
            </div>
            
            <div id="comparison" class="tabcontent">
                <h2>Encoder & Device Comparison</h2>
                <div style="margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                    <h3 style="margin-top: 0; color: #333;">Data Discovery & Reference Selection</h3>
                    <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                        <div>
                            <label for="reference-select" style="font-weight: bold; margin-right: 10px;">Reference:</label>
                            <select id="reference-select" onchange="updateReference()" style="padding: 8px; border: 1px solid #ccc; border-radius: 4px; min-width: 250px;">
                                <option value="">Select Reference...</option>
                            </select>
                        </div>
                        <div>
                            <button onclick="showDataDiscovery()" style="padding: 8px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                                📊 Data Discovery
                            </button>
                        </div>
                        <div>
                            <button onclick="resetReference()" style="padding: 8px 15px; background-color: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">
                                🔄 Reset
                            </button>
                        </div>
                    </div>
                    <div id="data-info" style="margin-top: 10px; font-size: 12px; color: #666;">
                        Click "Data Discovery" to explore available encoders, devices, and metrics
                    </div>
                </div>
                <div id="comparison-plot-container">
                {comparison_html}
                </div>
            </div>
            
            <div id="encoder-stats" class="tabcontent">
                <h2>Encoder Performance Analysis</h2>
                {encoder_stats_html}
            </div>
            
            <div id="errors" class="tabcontent">
                <h2>Error Analysis</h2>
                {error_html}
            </div>
            
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                // Global variables for data discovery
                var plotData = null;
                var availableReferences = [];
                
                // Reference selection functionality
                function updateReference() {{
                    var select = document.getElementById('reference-select');
                    var selectedReference = select.value;
                    var infoDiv = document.getElementById('data-info');
                    
                    if (selectedReference) {{
                        console.log('Selected reference:', selectedReference);
                        infoDiv.innerHTML = '<strong>Selected Reference:</strong> ' + selectedReference + '<br>' +
                                          '<em>Delta plots updated to show differences from this reference</em>';
                        
                        // Update comparison plots with new reference
                        updateComparisonPlots(selectedReference);
                    }} else {{
                        infoDiv.innerHTML = 'Click "Data Discovery" to explore available encoders, devices, and metrics';
                    }}
                }}
                
                // Data discovery interface
                function showDataDiscovery() {{
                    var infoDiv = document.getElementById('data-info');
                    
                    // Extract data from the existing plots
                    extractPlotData();
                    
                    if (availableReferences.length === 0) {{
                        infoDiv.innerHTML = '<span style="color: red;">No data found in comparison plots. Make sure quality data is loaded.</span>';
                        return;
                    }}
                    
                    var discoveryHtml = '<div style="background-color: white; padding: 15px; border: 1px solid #ddd; border-radius: 8px; margin-top: 10px;">';
                    discoveryHtml += '<h4 style="margin-top: 0;">📊 Available Data:</h4>';
                    
                    // Group by encoder
                    var encoders = {{}};
                    availableReferences.forEach(function(ref) {{
                        var parts = ref.split(' (');
                        var encoder = parts[0];
                        var device = parts[1] ? parts[1].replace(')', '') : 'Unknown';
                        
                        if (!encoders[encoder]) {{
                            encoders[encoder] = [];
                        }}
                        encoders[encoder].push(device);
                    }});
                    
                    discoveryHtml += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">';
                    
                    Object.keys(encoders).forEach(function(encoder) {{
                        discoveryHtml += '<div style="border: 1px solid #eee; padding: 10px; border-radius: 5px;">';
                        discoveryHtml += '<strong>' + encoder + '</strong><br>';
                        encoders[encoder].forEach(function(device) {{
                            var fullRef = encoder + ' (' + device + ')';
                            discoveryHtml += '<div style="margin: 5px 0; padding: 5px; background-color: #f8f9fa; border-radius: 3px;">';
                            discoveryHtml += '📱 ' + device;
                            discoveryHtml += ' <button onclick="selectReference(\\'' + fullRef + '\\')" style="margin-left: 10px; padding: 2px 8px; background-color: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 11px;">Select</button>';
                            discoveryHtml += '</div>';
                        }});
                        discoveryHtml += '</div>';
                    }});
                    
                    discoveryHtml += '</div>';
                    discoveryHtml += '<div style="margin-top: 10px; font-size: 12px; color: #666;">';
                    discoveryHtml += '<strong>Total combinations:</strong> ' + availableReferences.length + '<br>';
                    discoveryHtml += '<strong>Encoders:</strong> ' + Object.keys(encoders).length + '<br>';
                    discoveryHtml += '</div>';
                    discoveryHtml += '</div>';
                    
                    infoDiv.innerHTML = discoveryHtml;
                }}
                
                // Extract data from existing plots
                function extractPlotData() {{
                    availableReferences = [];
                    
                    // Try to extract from Plotly plots
                    var plotContainers = document.querySelectorAll('#comparison-plot-container .plotly-graph-div');
                    plotContainers.forEach(function(container) {{
                        if (container.data) {{
                            container.data.forEach(function(trace) {{
                                if (trace.name && trace.name.includes('(') && trace.name.includes(')')) {{
                                    // Extract encoder/device combinations
                                    if (!availableReferences.includes(trace.name)) {{
                                        availableReferences.push(trace.name);
                                    }}
                                }}
                            }});
                        }}
                    }});
                    
                    // Fallback: use hardcoded data if no plots found
                    if (availableReferences.length === 0) {{
                        availableReferences = [
                            'c2.dolby.encoder.hevc (V2413)',
                            'c2.qti.hevc.encoder.hdr (SM-S936U1)', 
                            'c2.google.av1.encoder (Pixel 8)'
                        ];
                    }}
                }}
                
                // Select reference from discovery interface
                function selectReference(reference) {{
                    var select = document.getElementById('reference-select');
                    select.value = reference;
                    updateReference();
                }}
                
                // Reset reference
                function resetReference() {{
                    var select = document.getElementById('reference-select');
                    var infoDiv = document.getElementById('data-info');
                    select.value = '';
                    infoDiv.innerHTML = 'Click "Data Discovery" to explore available encoders, devices, and metrics';
                }}
                
                // Global variable to store original plot data
                var originalPlotData = null;
                
                // Update comparison plots with new reference
                function updateComparisonPlots(reference) {{
                    console.log('Updating plots with reference:', reference);
                    
                    // Find the comparison plot container
                    var plotContainer = document.getElementById('comparison-plot-container');
                    var plotDivs = plotContainer.querySelectorAll('.plotly-graph-div');
                    
                    if (plotDivs.length === 0) {{
                        console.log('No comparison plots found to update');
                        return;
                    }}
                    
                    // Store original data if not already stored
                    if (!originalPlotData) {{
                        originalPlotData = plotDivs[0].data;
                        console.log('Stored original plot data:', originalPlotData);
                    }}
                    
                    // For each plot, update the reference
                    plotDivs.forEach(function(plotDiv) {{
                        if (plotDiv.data && plotDiv.layout) {{
                            // Update the plot data to reflect new reference
                            var newData = updatePlotDataWithNewReference(originalPlotData, reference);
                            if (newData) {{
                                Plotly.redraw(plotDiv, newData);
                            }}
                        }}
                    }});
                }}
                
                // Update plot data with new reference
                function updatePlotDataWithNewReference(plotData, newReference) {{
                    if (!plotData || plotData.length === 0) {{
                        console.log('No plot data available');
                        return null;
                    }}
                    
                    // Extract the encoder/device name from the reference string
                    var referenceName = newReference.replace(' Δ', '');
                    console.log('Looking for reference:', referenceName);
                    
                    var updatedData = [];
                    var referenceTrace = null;
                    
                    // Find the reference trace
                    plotData.forEach(function(trace) {{
                        if (trace.name && (trace.name.includes(referenceName) || trace.name.includes(newReference))) {{
                            referenceTrace = trace;
                            console.log('Found reference trace:', trace.name);
                        }}
                    }});
                    
                    if (!referenceTrace) {{
                        console.log('Reference trace not found for:', referenceName);
                        return null;
                    }}
                    
                    // Create new reference trace (zero line)
                    var newReferenceTrace = {{
                        x: referenceTrace.x,
                        y: new Array(referenceTrace.x.length).fill(0),
                        mode: referenceTrace.mode,
                        type: referenceTrace.type,
                        name: referenceName + ' (Reference)',
                        line: {{color: 'black', width: 3, dash: 'solid'}},
                        marker: referenceTrace.marker
                    }};
                    updatedData.push(newReferenceTrace);
                    
                    // Recalculate deltas for all other traces
                    plotData.forEach(function(trace) {{
                        if (trace !== referenceTrace && trace.name && !trace.name.includes('(Reference)')) {{
                            // Interpolate reference values at this trace's x points
                            var deltaValues = [];
                            for (var i = 0; i < trace.x.length; i++) {{
                                var xValue = trace.x[i];
                                var refValue = interpolateValue(referenceTrace.x, referenceTrace.y, xValue);
                                var delta = trace.y[i] - refValue;
                                deltaValues.push(delta);
                            }}
                            
                            var deltaTrace = {{
                                x: trace.x,
                                y: deltaValues,
                                mode: trace.mode,
                                type: trace.type,
                                name: trace.name.replace(' (Reference)', '') + ' Δ',
                                line: trace.line,
                                marker: trace.marker
                            }};
                            updatedData.push(deltaTrace);
                        }}
                    }});
                    
                    console.log('Updated plot data with', updatedData.length, 'traces');
                    return updatedData;
                }}
                
                // Simple linear interpolation function
                function interpolateValue(xArray, yArray, xValue) {{
                    if (xArray.length === 0) return 0;
                    if (xArray.length === 1) return yArray[0];
                    
                    // Find the two closest points
                    var closestIndex = 0;
                    var minDistance = Math.abs(xArray[0] - xValue);
                    
                    for (var i = 1; i < xArray.length; i++) {{
                        var distance = Math.abs(xArray[i] - xValue);
                        if (distance < minDistance) {{
                            minDistance = distance;
                            closestIndex = i;
                        }}
                    }}
                    
                    // If exact match or at boundary
                    if (minDistance < 1e-6 || closestIndex === 0 || closestIndex === xArray.length - 1) {{
                        return yArray[closestIndex];
                    }}
                    
                    // Linear interpolation
                    var i1 = closestIndex;
                    var i2 = xArray[closestIndex] > xValue ? closestIndex - 1 : closestIndex + 1;
                    
                    if (i2 < 0 || i2 >= xArray.length) {{
                        return yArray[closestIndex];
                    }}
                    
                    var x1 = xArray[i1], y1 = yArray[i1];
                    var x2 = xArray[i2], y2 = yArray[i2];
                    
                    if (Math.abs(x2 - x1) < 1e-6) {{
                        return y1;
                    }}
                    
                    return y1 + (y2 - y1) * (xValue - x1) / (x2 - x1);
                }}
                
                // Populate reference dropdown with available encoders/devices
                function populateReferenceDropdown() {{
                    var select = document.getElementById('reference-select');
                    
                    // Extract data first
                    extractPlotData();
                    
                    // Clear existing options except the first one
                    while (select.children.length > 1) {{
                        select.removeChild(select.lastChild);
                    }}
                    
                    availableReferences.forEach(function(ref) {{
                        var option = document.createElement('option');
                        option.value = ref;
                        option.textContent = ref;
                        select.appendChild(option);
                    }});
                }}
                
                // Initialize when page loads
                document.addEventListener('DOMContentLoaded', function() {{
                    populateReferenceDropdown();
                    initializeOriginalData();
                }});
                
                // Initialize original plot data
                function initializeOriginalData() {{
                    setTimeout(function() {{
                        var plotContainer = document.getElementById('comparison-plot-container');
                        var plotDivs = plotContainer.querySelectorAll('.plotly-graph-div');
                        
                        if (plotDivs.length > 0 && !originalPlotData) {{
                            originalPlotData = plotDivs[0].data;
                            console.log('Initialized original plot data:', originalPlotData);
                        }}
                    }}, 1000); // Wait 1 second for plots to load
                }}
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def create_encoder_statistics_plots(self, stats_files: List[str]) -> go.Figure:
        """Create aggregated time series plots from encoder statistics CSV files"""
        if not PLOTLY_AVAILABLE or not stats_files:
            return None
            
        try:
            import pandas as pd
            import numpy as np
            
            # Load all encoding data CSV files
            all_data = []
            for file in stats_files:
                if file.endswith('_encoding_data.csv') and os.path.exists(file):
                    df = pd.read_csv(file)
                    if not df.empty:
                        all_data.append(df)
            
            if not all_data:
                self.logger.warning("No encoding data CSV files found")
                return None
                
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Loaded encoder statistics data: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            
            # Create subplots for aggregated metrics
        fig = make_subplots(
            rows=2, cols=2,
                subplot_titles=("Encoding Latency (Aggregated)", "Processing Framerate (Aggregated)", 
                              "Pipeline Depth Over Time", "Bitrate Variability"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
            # Plot 1: Aggregated Encoding Latency (like your example)
            if 'proctime' in combined_df.columns and 'rel_pts' in combined_df.columns:
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec].sort_values('rel_pts')
                    if len(codec_data) > 0:
                        # Convert proctime from nanoseconds to milliseconds
                        latency_ms = (codec_data['proctime'] / 1000000).tolist()
                        time_sec = (codec_data['rel_pts'] / 1000000).tolist()  # Convert to seconds
                        
                        # Calculate aggregated statistics
                        mean_latency = np.mean(latency_ms)
                        p50 = np.percentile(latency_ms, 50)
                        p95 = np.percentile(latency_ms, 95)
                        p99 = np.percentile(latency_ms, 99)
                        
                        # Create rolling mean and confidence interval
                        window_size = max(1, len(latency_ms) // 20)  # Adaptive window size
                        rolling_mean = pd.Series(latency_ms).rolling(window=window_size, center=True).mean()
                        rolling_std = pd.Series(latency_ms).rolling(window=window_size, center=True).std()
                        
                        # Add main line (rolling mean)
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec,
                                y=rolling_mean.tolist(),
                                mode='lines',
                                name=f'{codec} - Mean: {mean_latency:.2f}ms, p50,p95,p99: {p50:.0f}, {p95:.0f}, {p99:.0f}',
                                line=dict(color='blue', width=3),
                                fill='tonexty',
                                fillcolor='rgba(0,100,200,0.2)'
                            ),
                            row=1, col=1
                        )
                        
                        # Add confidence interval (rolling mean ± 1 std)
                        upper_bound = (rolling_mean + rolling_std).tolist()
                        lower_bound = (rolling_mean - rolling_std).tolist()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec + time_sec[::-1],  # x coordinates for filled area
                                y=upper_bound + lower_bound[::-1],  # y coordinates for filled area
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(0,100,200,0.1)',
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False,
                                hoverinfo="skip"
                            ),
                            row=1, col=1
                        )
            
            # Plot 2: Aggregated Processing Framerate (like your example)
            if 'proc_fps' in combined_df.columns and 'inflight' in combined_df.columns and 'rel_pts' in combined_df.columns:
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec].sort_values('rel_pts')
                    if len(codec_data) > 0:
                        proc_fps = codec_data['proc_fps'].tolist()
                        inflight = codec_data['inflight'].tolist()
                        time_sec = (codec_data['rel_pts'] / 1000000).tolist()
                        
                        # Calculate aggregated statistics
                        mean_fps = np.mean(proc_fps)
                        
                        # Create rolling mean and confidence interval
                        window_size = max(1, len(proc_fps) // 20)
                        rolling_mean_fps = pd.Series(proc_fps).rolling(window=window_size, center=True).mean()
                        rolling_std_fps = pd.Series(proc_fps).rolling(window=window_size, center=True).std()
                        
                        # Add main line (rolling mean)
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec,
                                y=rolling_mean_fps.tolist(),
                                mode='lines',
                                name=f'{codec} - {mean_fps:.1f} fps',
                                line=dict(color='blue', width=3),
                                fill='tonexty',
                                fillcolor='rgba(0,100,200,0.2)'
                            ),
                            row=1, col=2
                        )
                        
                        # Add confidence interval
                        upper_bound_fps = (rolling_mean_fps + rolling_std_fps).tolist()
                        lower_bound_fps = (rolling_mean_fps - rolling_std_fps).tolist()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec + time_sec[::-1],
                                y=upper_bound_fps + lower_bound_fps[::-1],
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(0,100,200,0.1)',
                                line=dict(color='rgba(255,255,255,0)'),
                                showlegend=False,
                                hoverinfo="skip"
                            ),
                            row=1, col=2
                        )
            
            # Plot 3: Pipeline Depth Over Time (Inflight frames)
            if 'inflight' in combined_df.columns and 'rel_pts' in combined_df.columns:
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec].sort_values('rel_pts')
                    if len(codec_data) > 0:
                        inflight = codec_data['inflight'].tolist()
                        time_sec = (codec_data['rel_pts'] / 1000000).tolist()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec,
                                y=inflight,
                                mode='lines+markers',
                                name=f'Pipeline Depth - {codec}',
                                line=dict(width=2)
                            ),
                            row=2, col=1
                        )
            
            # Plot 4: Bitrate Variability
            if 'bitrate_per_frame_bps' in combined_df.columns and 'rel_pts' in combined_df.columns:
                for codec in combined_df['codec'].unique():
                    codec_data = combined_df[combined_df['codec'] == codec].sort_values('rel_pts')
                    if len(codec_data) > 0:
                        bitrate_kbps = (codec_data['bitrate_per_frame_bps'] / 1000).tolist()
                        time_sec = (codec_data['rel_pts'] / 1000000).tolist()
                        
                        fig.add_trace(
                            go.Scatter(
                                x=time_sec,
                                y=bitrate_kbps,
                                mode='lines+markers',
                                name=f'Bitrate/Frame - {codec}',
                                line=dict(width=2)
                            ),
                            row=2, col=2
                        )
            
            # Update layout
            fig.update_layout(
                title="Encoder Performance Time Series Analysis",
                height=1200,  # Increased from 800 to make graphs bigger
                showlegend=True,
                # Enable interactive features
                autosize=True,
                # Make the plot more interactive
                dragmode='zoom',
                # Add hover information
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
            
        except Exception as e:
            self.logger.error(f"Error creating encoder statistics plots: {e}")
            return None

    def create_encoder_aggregated_stats(self, stats_files: List[str]) -> Dict[str, Any]:
        """Create aggregated statistics from encoder statistics CSV files"""
        try:
            import pandas as pd
            
            # Load all encoding data CSV files
            all_data = []
            for file in stats_files:
                if file.endswith('_encoding_data.csv') and os.path.exists(file):
                    df = pd.read_csv(file)
                    all_data.append(df)
            
            if not all_data:
                self.logger.warning("No encoding data CSV files found for aggregated stats")
                return {}
                
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Calculate aggregated statistics by codec
            stats_by_codec = {}
            
            for codec in combined_df['codec'].unique():
                codec_data = combined_df[combined_df['codec'] == codec]
                
                stats = {
                    'codec': codec,
                    'total_frames': len(codec_data),
                    'avg_encoding_time_ms': codec_data['proctime'].mean() if 'proctime' in codec_data.columns else None,
                    'median_encoding_time_ms': codec_data['proctime'].median() if 'proctime' in codec_data.columns else None,
                    'p95_encoding_time_ms': codec_data['proctime'].quantile(0.95) if 'proctime' in codec_data.columns else None,
                    'p99_encoding_time_ms': codec_data['proctime'].quantile(0.99) if 'proctime' in codec_data.columns else None,
                    'avg_pipeline_depth': codec_data['inflight'].mean() if 'inflight' in codec_data.columns else None,
                    'max_pipeline_depth': codec_data['inflight'].max() if 'inflight' in codec_data.columns else None,
                    'avg_bitrate_per_frame_kbps': (codec_data['bitrate_per_frame_bps'].mean() / 1000) if 'bitrate_per_frame_bps' in codec_data.columns else None,
                    'avg_processing_fps': codec_data['proc_fps'].mean() if 'proc_fps' in codec_data.columns else None,
                    'min_processing_fps': codec_data['proc_fps'].min() if 'proc_fps' in codec_data.columns else None,
                    'max_processing_fps': codec_data['proc_fps'].max() if 'proc_fps' in codec_data.columns else None,
                }
                
                # Calculate latency statistics if timing fields are available
                if all(col in codec_data.columns for col in ['starttime', 'stoptime']):
                    codec_data['latency_ms'] = (codec_data['stoptime'] - codec_data['starttime']) * 1000  # Convert to ms
                    stats.update({
                        'avg_latency_ms': codec_data['latency_ms'].mean(),
                        'median_latency_ms': codec_data['latency_ms'].median(),
                        'p95_latency_ms': codec_data['latency_ms'].quantile(0.95),
                        'p99_latency_ms': codec_data['latency_ms'].quantile(0.99),
                    })
                
                stats_by_codec[codec] = stats
            
            return stats_by_codec
            
        except Exception as e:
            self.logger.error(f"Error creating aggregated encoder statistics: {e}")
            return {}

    def create_comparison_plots(self, quality_data: Dict[str, Any]) -> go.Figure:
        """Create delta comparison plots against a reference encoder/device"""
        if not PLOTLY_AVAILABLE:
            return None
            
        # Create subplots - simplified layout with 2 rows, 1 column for cleaner view
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("VMAF Delta vs Reference", "SI/TI Complexity Analysis"),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]],
            vertical_spacing=0.15  # More space between plots
        )
        
        # VMAF Delta comparison - choose first encoder as reference
        if quality_data["vmaf_data"]:
            try:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
                self.logger.info(f"VMAF data shape: {vmaf_df.shape}, columns: {vmaf_df.columns.tolist()}")
                
                # Choose reference encoder-device combination (first one alphabetically)
                # Create encoder-device combinations
                vmaf_df["encoder_device"] = vmaf_df["encoder"] + " (" + vmaf_df["model"] + ")"
                encoder_devices = sorted(vmaf_df["encoder_device"].unique())
                if not encoder_devices:
                    return None
                    
                reference_encoder_device = encoder_devices[0]
                self.logger.info(f"Using {reference_encoder_device} as reference")
                
                # Create reference curve by interpolating at common bitrate points
                reference_data = vmaf_df[vmaf_df["encoder_device"] == reference_encoder_device].sort_values("bitrate")
                if len(reference_data) < 2:
                    return None
                    
                # Interpolate reference curve
                import numpy as np
                reference_bitrates = reference_data["bitrate"].values
                reference_vmaf = reference_data["vmaf"].values
                
                for encoder_device in encoder_devices:
                    encoder_device_data = vmaf_df[vmaf_df["encoder_device"] == encoder_device].sort_values("bitrate")
                    if encoder_device == reference_encoder_device:
                        # Show reference as zero line
                fig.add_trace(
                    go.Scatter(
                                x=(encoder_device_data["bitrate"] / 1000).tolist(),
                                y=[0] * len(encoder_device_data),
                                mode='lines+markers',
                                name=f'{encoder_device} (Reference)',
                                line=dict(width=3, color='black', dash='solid')
                            ),
                            row=1, col=1
                        )
                    else:
                        # Calculate delta against reference
                        delta_values = []
                        for _, row in encoder_device_data.iterrows():
                            # Interpolate reference VMAF at this bitrate
                            ref_vmaf = np.interp(row["bitrate"], reference_bitrates, reference_vmaf)
                            delta = row["vmaf"] - ref_vmaf
                            delta_values.append(delta)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=(encoder_device_data["bitrate"] / 1000).tolist(),
                                y=delta_values,
                                mode='lines+markers',
                                name=f'{encoder_device} Δ',
                        line=dict(width=3)
                    ),
                    row=1, col=1
                )
            except Exception as e:
                self.logger.error(f"Error in VMAF delta comparison plot: {e}")
                self.logger.error(f"VMAF data: {quality_data['vmaf_data'][:2] if quality_data['vmaf_data'] else 'No data'}")
        
        # PSNR Delta comparison - choose first device as reference
        if quality_data["psnr_data"]:
            try:
            psnr_df = pd.DataFrame(quality_data["psnr_data"])
                self.logger.info(f"PSNR data shape: {psnr_df.shape}, columns: {psnr_df.columns.tolist()}")
            for device in psnr_df["device"].unique():
                device_data = psnr_df[psnr_df["device"] == device]
                    self.logger.info(f"Device {device} has {len(device_data)} data points")
                fig.add_trace(
                    go.Scatter(
                            x=(device_data["bitrate"] / 1000).tolist(),  # Convert bps to kbps and ensure list
                            y=device_data["psnr"].tolist(),
                        mode='markers+lines',
                        name=f'Device {device}',
                        line=dict(width=3)
                    ),
                    row=1, col=2
                )
            except Exception as e:
                self.logger.error(f"Error in device comparison plot: {e}")
                self.logger.error(f"PSNR data: {quality_data['psnr_data'][:2] if quality_data['psnr_data'] else 'No data'}")
        
        # SI/TI Complexity analysis - plot SI vs TI for different sources
        if quality_data["vmaf_data"]:
            try:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
                # Check if we have SI/TI data from the --siti analysis
                if all(col in vmaf_df.columns for col in ['si_avg', 'ti_avg', 'source']):
                    # Group by source to get ONE point per source (SI/TI values are constant per source)
                    complexity_data = vmaf_df.groupby("source").agg({
                        "si_avg": "first",  # Use first value since SI/TI is constant per source
                        "ti_avg": "first",  # Use first value since SI/TI is constant per source
                        "vmaf": "mean"      # Average VMAF across all tests for this source
                    }).reset_index()
                    
                    # Clean up source names for display
                    complexity_data["source_name"] = complexity_data["source"].apply(
                        lambda x: os.path.basename(str(x)) if x else "Unknown"
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=complexity_data["ti_avg"].tolist(),  # TI on x-axis (temporal complexity)
                            y=complexity_data["si_avg"].tolist(),  # SI on y-axis (spatial complexity)
                            mode='markers',  # Remove text labels - just markers
                            name="SI/TI Complexity",
                            marker=dict(
                                size=25,  # Slightly larger for visibility without text
                                color='blue',  # Single color since we don't need color scale
                                line=dict(width=2, color='black')  # Black outline for visibility
                            ),
                            hovertemplate="<b>Source: %{customdata}</b><br>" +
                                        "TI: %{x:.2f}<br>" +
                                        "SI: %{y:.2f}<br>" +
                                        "<extra></extra>",
                            customdata=complexity_data["source_name"].tolist()  # Include filename in hover only
                        ),
                        row=2, col=1
                    )
                else:
                    self.logger.warning("No SI/TI complexity data found. Make sure --siti flag is used with encapp_quality")
                    # Fallback to basic source analysis
                    if "source" in vmaf_df.columns:
            source_complexity = vmaf_df.groupby("source").agg({
                "vmaf": "mean",
                "bitrate": "mean"
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                                x=(source_complexity["bitrate"] / 1000).tolist(),
                                y=source_complexity["vmaf"].tolist(),
                    mode='markers',
                                text=source_complexity["source"].tolist(),
                                name="Source Complexity (Fallback)",
                    marker=dict(size=10, color='red')
                ),
                row=2, col=1
            )
            except Exception as e:
                self.logger.error(f"Error in SI/TI complexity plot: {e}")
        
        # Bitrate efficiency (VMAF per bitrate)
        if quality_data["vmaf_data"]:
            try:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
            vmaf_df["efficiency"] = vmaf_df["vmaf"] / vmaf_df["bitrate"]
            
            for encoder in vmaf_df["encoder"].unique():
                encoder_data = vmaf_df[vmaf_df["encoder"] == encoder]
                fig.add_trace(
                    go.Scatter(
                            x=(encoder_data["bitrate"] / 1000).tolist(),  # Convert bps to kbps and ensure list
                            y=encoder_data["efficiency"].tolist(),
                        mode='markers+lines',
                        name=f'Efficiency - {encoder}',
                        line=dict(dash='dash')
                    ),
                    row=2, col=2
                )
            except Exception as e:
                self.logger.error(f"Error in bitrate efficiency plot: {e}")
        
        # Update layout
        fig.update_layout(
            title="Codec Performance Comparison",
            height=800,
            showlegend=True,
            # Enable interactive features
            autosize=True,
            dragmode='zoom',
            hovermode='x unified'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Bitrate (kbps)", row=1, col=1)
        fig.update_yaxes(title_text="VMAF Delta", row=1, col=1)
        fig.update_xaxes(title_text="Temporal Information (TI)", row=2, col=1)
        fig.update_yaxes(title_text="Spatial Information (SI)", row=2, col=1)
        
        return fig
    
    def _create_aggregated_performance_stats(self, test_results: List[TestResult], quality_data: Dict[str, Any]) -> str:
        """Create detailed encoder-device performance analysis with latency and quality metrics"""
        if not PLOTLY_AVAILABLE:
            return """
            <div style="text-align: center; padding: 50px; color: #666;">
                <h3>Plotly Not Available</h3>
                <p>Cannot generate performance statistics without Plotly library.</p>
            </div>
            """
        
        try:
            import pandas as pd
            
            # Extract detailed performance data from quality metrics
            performance_data = []
            for result in test_results:
                if result.success and result.quality_metrics:
                    for metrics in result.quality_metrics:
                        # Calculate derived metrics
                        duration_sec = metrics.get("duration", 0)
                        framecount = metrics.get("framecount", 0)
                        
                        # Calculate latency metrics (approximate)
                        if duration_sec > 0 and framecount > 0:
                            avg_frame_time = duration_sec / framecount * 1000  # ms per frame
                            fps = framecount / duration_sec
                        else:
                            avg_frame_time = 0
                            fps = 0
                        
                        # Calculate bitrate accuracy
                        target_bitrate = metrics.get("bitrate_bps", 0)
                        actual_bitrate = metrics.get("calculated_bitrate_bps", 0)
                        bitrate_accuracy = ((actual_bitrate - target_bitrate) / target_bitrate * 100) if target_bitrate > 0 else 0
                        
                        performance_data.append({
                            "encoder": metrics.get("codec", result.encoder),
                            "device": result.device_serial,
                            "model": metrics.get("model", result.device_serial),
                            "target_bitrate": target_bitrate,
                            "actual_bitrate": actual_bitrate,
                            "bitrate_accuracy": bitrate_accuracy,
                            "vmaf": metrics.get("vmaf", 0),
                            "psnr": metrics.get("psnr", 0),
                            "ssim": metrics.get("ssim", 0),
                            "avg_frame_time_ms": avg_frame_time,
                            "fps": fps,
                            "duration_sec": duration_sec,
                            "framecount": framecount,
                            "test_name": result.test_name
                        })
            
            if not performance_data:
                return """
                <div style="text-align: center; padding: 50px; color: #666;">
                    <h3>No Performance Data Available</h3>
                    <p>No quality metrics found to generate performance analysis.</p>
                </div>
                """
            
            df = pd.DataFrame(performance_data)
            
            # Create detailed performance plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Encoder-Device Performance Matrix", "Bitrate Accuracy Analysis", 
                              "Frame Processing Latency", "Quality vs Bitrate Efficiency"),
                specs=[[{"type": "bar", "secondary_y": False}, {"type": "bar", "secondary_y": False}],
                       [{"type": "scatter", "secondary_y": False}, {"type": "scatter", "secondary_y": False}]]
            )
            
            # 1. Encoder-Device Performance Matrix (VMAF)
            encoder_device_vmaf = df.groupby(["encoder", "model"]).agg({
                "vmaf": ["mean", "std", "count"]
            }).round(2)
            
            # Flatten column names
            encoder_device_vmaf.columns = ["vmaf_mean", "vmaf_std", "vmaf_count"]
            encoder_device_vmaf = encoder_device_vmaf.reset_index()
            
            # Create encoder-device labels
            encoder_device_vmaf["encoder_device"] = encoder_device_vmaf["encoder"].str.split(".").str[-1] + " (" + encoder_device_vmaf["model"] + ")"
            
            fig.add_trace(
                go.Bar(
                    x=encoder_device_vmaf["encoder_device"].tolist(),
                    y=encoder_device_vmaf["vmaf_mean"].tolist(),
                    name="Avg VMAF",
                    marker_color='lightblue',
                    text=encoder_device_vmaf["vmaf_mean"].round(1).tolist(),
                    textposition='auto',
                    error_y=dict(type='data', array=encoder_device_vmaf["vmaf_std"].tolist())
                ),
                row=1, col=1
            )
            
            # 2. Bitrate Accuracy Analysis
            encoder_device_accuracy = df.groupby(["encoder", "model"]).agg({
                "bitrate_accuracy": ["mean", "std", "count"]
            }).round(2)
            
            encoder_device_accuracy.columns = ["accuracy_mean", "accuracy_std", "accuracy_count"]
            encoder_device_accuracy = encoder_device_accuracy.reset_index()
            encoder_device_accuracy["encoder_device"] = encoder_device_accuracy["encoder"].str.split(".").str[-1] + " (" + encoder_device_accuracy["model"] + ")"
            
            fig.add_trace(
                go.Bar(
                    x=encoder_device_accuracy["encoder_device"].tolist(),
                    y=encoder_device_accuracy["accuracy_mean"].tolist(),
                    name="Bitrate Accuracy (%)",
                    marker_color='orange',
                    text=encoder_device_accuracy["accuracy_mean"].round(1).tolist(),
                    textposition='auto',
                    error_y=dict(type='data', array=encoder_device_accuracy["accuracy_std"].tolist())
                ),
                row=1, col=2
            )
            
            # 3. Frame Processing Latency (scatter plot)
            for encoder in df["encoder"].unique():
                encoder_data = df[df["encoder"] == encoder]
                for model in encoder_data["model"].unique():
                    model_data = encoder_data[encoder_data["model"] == model]
                    
                    encoder_short = encoder.split(".")[-1] if "." in encoder else encoder
                    display_name = f"{encoder_short} ({model})"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=model_data["target_bitrate"].tolist(),
                            y=model_data["avg_frame_time_ms"].tolist(),
                            mode='markers+lines',
                            name=display_name,
                            marker=dict(size=8),
                            line=dict(width=2)
                        ),
                        row=2, col=1
                    )
            
            # 4. Quality vs Bitrate Efficiency (VMAF per kbps)
            for encoder in df["encoder"].unique():
                encoder_data = df[df["encoder"] == encoder]
                for model in encoder_data["model"].unique():
                    model_data = encoder_data[encoder_data["model"] == model]
                    
                    encoder_short = encoder.split(".")[-1] if "." in encoder else encoder
                    display_name = f"{encoder_short} ({model})"
                    
                    # Calculate efficiency (VMAF per kbps)
                    efficiency = (model_data["vmaf"] / (model_data["actual_bitrate"] / 1000)).tolist()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=(model_data["actual_bitrate"] / 1000).tolist(),  # Convert to kbps
                            y=efficiency,
                            mode='markers+lines',
                            name=display_name,
                            marker=dict(size=8),
                            line=dict(width=2)
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title="Detailed Encoder-Device Performance Analysis",
                height=1000,
                showlegend=True,
                autosize=True,
                dragmode='zoom',
                hovermode='x unified'
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Encoder-Device Combination", row=1, col=1)
            fig.update_yaxes(title_text="Average VMAF", row=1, col=1)
            fig.update_xaxes(title_text="Encoder-Device Combination", row=1, col=2)
            fig.update_yaxes(title_text="Bitrate Accuracy (%)", row=1, col=2)
            fig.update_xaxes(title_text="Target Bitrate (bps)", row=2, col=1)
            fig.update_yaxes(title_text="Frame Processing Time (ms)", row=2, col=1)
            fig.update_xaxes(title_text="Actual Bitrate (kbps)", row=2, col=2)
            fig.update_yaxes(title_text="Efficiency (VMAF/kbps)", row=2, col=2)
            
            # Convert to HTML
            from plotly.offline import plot
            return plot(fig, output_type='div', include_plotlyjs=False)
            
        except Exception as e:
            self.logger.error(f"Error creating detailed performance stats: {e}")
            return f"""
            <div style="text-align: center; padding: 50px; color: #666;">
                <h3>Error Generating Performance Statistics</h3>
                <p>Could not generate detailed performance statistics: {str(e)}</p>
                <p>This may be due to insufficient quality data or plotting library issues.</p>
            </div>
            """
    
    def _create_summary_plots(self, test_results: List[TestResult]) -> go.Figure:
        """Create summary plots for test results"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Test Success Rate", "Codecs Tested", 
                          "Test Duration Distribution", "Device Performance"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Test Success Rate (Pie Chart)
        success_count = sum(1 for r in test_results if r.success)
        failure_count = len(test_results) - success_count
        fig.add_trace(
            go.Pie(
                labels=["Success", "Failure"],
                values=[success_count, failure_count],
                name="Test Results"
            ),
            row=1, col=1
        )
        
        # Codecs Tested (show number of tests per codec)
        encoder_stats = {}
        for result in test_results:
            encoder = result.encoder
            if encoder not in encoder_stats:
                encoder_stats[encoder] = 0
            encoder_stats[encoder] += 1
        
        encoders = list(encoder_stats.keys())
        test_counts = [encoder_stats[enc] for enc in encoders]
        
        # Shorten encoder names for better display
        short_encoders = []
        for enc in encoders:
            if "." in enc:
                # Take the last part after the last dot
                short_enc = enc.split(".")[-1]
            else:
                short_enc = enc
            short_encoders.append(short_enc)
        
        fig.add_trace(
            go.Bar(
                x=short_encoders,
                y=test_counts,
                name="Tests per Codec",
                marker_color='lightblue',
                text=test_counts,
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Test Duration Distribution
        durations = [r.duration for r in test_results if r.success]
        fig.add_trace(
            go.Histogram(
                x=durations,
                name="Duration Distribution",
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Device Performance (show number of tests per device)
        device_stats = {}
        device_models = {}
        for result in test_results:
            device = result.device_serial
            if device not in device_stats:
                device_stats[device] = 0
                # Try to get model name from quality metrics if available
                model = device  # Default to serial
                if result.quality_metrics:
                    for metrics in result.quality_metrics:
                        if metrics.get("model") and metrics.get("model") != device:
                            model = metrics.get("model")
                            break
                device_models[device] = model
            device_stats[device] += 1
        
        devices = list(device_stats.keys())
        test_counts = [device_stats[dev] for dev in devices]
        device_labels = [device_models.get(dev, dev) for dev in devices]
        
        fig.add_trace(
            go.Bar(
                x=device_labels,
                y=test_counts,
                name="Tests per Device",
                marker_color='orange',
                text=test_counts,
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Test Summary Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_error_analysis_plots(self, test_results: List[TestResult]) -> go.Figure:
        """Create error analysis plots for failed tests"""
        if not PLOTLY_AVAILABLE:
            return None
            
        failed_results = [r for r in test_results if not r.success]
        if not failed_results:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No failed tests to analyze",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Error Analysis - No Failures")
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Error Types", "Failed Tests by Encoder", 
                          "Failed Tests by Device", "Error Timeline"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Error Types
        error_types = {}
        for result in failed_results:
            error = result.error_message or "Unknown Error"
            error_types[error] = error_types.get(error, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(error_types.keys()),
                values=list(error_types.values()),
                name="Error Types"
            ),
            row=1, col=1
        )
        
        # Failed tests by encoder
        encoder_failures = {}
        for result in failed_results:
            encoder = result.encoder
            encoder_failures[encoder] = encoder_failures.get(encoder, 0) + 1
        
        fig.add_trace(
            go.Bar(
                x=list(encoder_failures.keys()),
                y=list(encoder_failures.values()),
                name="Failures by Encoder",
                marker_color='red'
            ),
            row=1, col=2
        )
        
        # Failed tests by device
        device_failures = {}
        for result in failed_results:
            device = result.device_serial
            device_failures[device] = device_failures.get(device, 0) + 1
        
            fig.add_trace(
                go.Bar(
                x=list(device_failures.keys()),
                y=list(device_failures.values()),
                name="Failures by Device",
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Error timeline (if we had timestamps)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(failed_results))),
                y=[1] * len(failed_results),
                mode='markers',
                name="Failed Tests",
                marker=dict(size=10, color='red')
            ),
            row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Error Analysis Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def generate_static_report(self, test_results: List[TestResult], 
                             quality_results: List[Dict[str, Any]] = None) -> str:
        """Generate static text/HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"ava_static_report_{timestamp}.html"
        
        html_content = self._generate_html_report(test_results, quality_results)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Static report saved to: {report_file}")
        return str(report_file)
    
    def generate_json_report(self, test_results: List[TestResult], 
                           quality_results: List[Dict[str, Any]] = None) -> str:
        """Generate JSON report for programmatic access"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"ava_report_{timestamp}.json"
        
        report_data = {
            "timestamp": timestamp,
            "summary": self._generate_summary(test_results, quality_results),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "device_serial": r.device_serial,
                    "encoder": r.encoder,
                    "success": r.success,
                    "duration": r.duration,
                    "quality_metrics": r.quality_metrics,
                    "error_message": r.error_message
                }
                for r in test_results
            ],
            "quality_results": quality_results or []
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"JSON report saved to: {report_file}")
        return str(report_file)
    
    def _generate_summary(self, test_results: List[TestResult], 
                         quality_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Get unique encoders and devices
        encoders = set(r.encoder for r in test_results)
        devices = set(r.device_serial for r in test_results)
        
        # Calculate average duration
        durations = [r.duration for r in test_results if r.success]
        avg_duration = sum(durations) / len(durations) if durations else 0
        total_duration = sum(durations)
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "unique_encoders": len(encoders),
            "unique_devices": len(devices),
            "total_duration": total_duration,
            "avg_duration": avg_duration
        }
    
    def _generate_html_report(self, test_results: List[TestResult], 
                            quality_results: List[Dict[str, Any]] = None) -> str:
        """Generate basic HTML report"""
        summary = self._generate_summary(test_results, quality_results)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AVA Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .success {{ border-left-color: #4CAF50; }}
                .failure {{ border-left-color: #f44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>AVA Test Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {summary['total_tests']}</p>
                <p>Successful: {summary['successful_tests']}</p>
                <p>Failed: {summary['failed_tests']}</p>
                <p>Success Rate: {summary['success_rate']:.1%}</p>
                <p>Unique Encoders: {summary['unique_encoders']}</p>
                <p>Unique Devices: {summary['unique_devices']}</p>
                <p>Total Duration: {summary['total_duration']:.2f}s</p>
                <p>Average Duration: {summary['avg_duration']:.2f}s</p>
            </div>
        
                <h2>Test Results</h2>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Device</th>
                        <th>Encoder</th>
                        <th>Status</th>
                        <th>Duration (s)</th>
                        <th>Error Message</th>
                    </tr>
        """
        
        for result in test_results:
            status = "Success" if result.success else "Failed"
            status_class = "success" if result.success else "failure"
            error_msg = result.error_message or ""
            
            html_content += f"""
                <tr class="{status_class}">
                        <td>{result.test_name}</td>
                        <td>{result.device_serial}</td>
                        <td>{result.encoder}</td>
                    <td>{status}</td>
                        <td>{result.duration:.2f}</td>
                        <td>{error_msg}</td>
                    </tr>
            """
        
        html_content += """
                </table>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_basic_html_report(self, test_results: List[TestResult]) -> str:
        """Generate basic HTML report when Plotly is not available"""
        return self._generate_html_report(test_results)
    
    def generate_comprehensive_report(self, test_results: List[TestResult], 
                                    quality_results: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """Generate all types of reports"""
        reports = {}
        
        try:
            reports["interactive"] = self.generate_interactive_report(test_results, quality_results)
        except Exception as e:
            self.logger.error(f"Failed to generate interactive report: {e}")
            reports["interactive"] = None
        
        try:
            reports["static"] = self.generate_static_report(test_results, quality_results)
        except Exception as e:
            self.logger.error(f"Failed to generate static report: {e}")
            reports["static"] = None
        
        try:
            reports["json"] = self.generate_json_report(test_results, quality_results)
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            reports["json"] = None
        
        return reports


if __name__ == "__main__":
    # Example usage
    report_generator = ReportGenerator(debug=True)
    
    # Example test results
    test_results = [
        TestResult("test_bitrate_mode", "device1", "c2.qti.hevc.encoder", True, 10.5),
        TestResult("test_key_frame_interval", "device1", "c2.qti.hevc.encoder", True, 8.2),
        TestResult("test_chroma_offset", "device1", "c2.qti.hevc.encoder", False, 5.1, 
                  error_message="Test failed due to encoding error")
    ]
    
    # Example quality results
    quality_results = [
        {"test_name": "test_bitrate_mode", "psnr": 35.2, "ssim": 0.95, "vmaf": 85.3, "bitrate": 1000000},
        {"test_name": "test_key_frame_interval", "psnr": 32.8, "ssim": 0.92, "vmaf": 82.1, "bitrate": 800000}
    ]
    
    # Generate reports
    reports = report_generator.generate_comprehensive_report(test_results, quality_results)
    
    print("Generated reports:")
    for report_type, report_path in reports.items():
        if report_path:
            print(f"  {report_type}: {report_path}")
        else:
            print(f"  {report_type}: Failed to generate")