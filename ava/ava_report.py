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
                
            # Extract quality metrics
            metrics = result.quality_metrics
            if "vmaf" in metrics:
                quality_data["vmaf_data"].append({
                    "encoder": result.encoder,
                    "device": result.device_serial,
                    "bitrate": result.bitrate,
                    "source": result.source_file,
                    "resolution": result.resolution,
                    "framerate": result.framerate,
                    "vmaf": metrics["vmaf"],
                    "test_name": result.test_name
                })
            
            if "psnr" in metrics:
                quality_data["psnr_data"].append({
                    "encoder": result.encoder,
                    "device": result.device_serial,
                    "bitrate": result.bitrate,
                    "source": result.source_file,
                    "resolution": result.resolution,
                    "framerate": result.framerate,
                    "psnr": metrics["psnr"],
                    "test_name": result.test_name
                })
            
            if "ssim" in metrics:
                quality_data["ssim_data"].append({
                    "encoder": result.encoder,
                    "device": result.device_serial,
                    "bitrate": result.bitrate,
                    "source": result.source_file,
                    "resolution": result.resolution,
                    "framerate": result.framerate,
                    "ssim": metrics["ssim"],
                    "test_name": result.test_name
                })
            
            if result.qp_values:
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
    
    def create_quality_plots(self, quality_data: Dict[str, Any]) -> go.Figure:
        """Create interactive quality metrics plots"""
        if not PLOTLY_AVAILABLE:
            return None
            
        # Create subplots for different quality metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("VMAF vs Bitrate", "PSNR vs Bitrate", 
                          "SSIM vs Bitrate", "QP Statistics"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # VMAF plot
        if quality_data["vmaf_data"]:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
            for encoder in vmaf_df["encoder"].unique():
                encoder_data = vmaf_df[vmaf_df["encoder"] == encoder]
                fig.add_trace(
                    go.Scatter(
                        x=encoder_data["bitrate"],
                        y=encoder_data["vmaf"],
                        mode='markers+lines',
                        name=f'VMAF - {encoder}',
                        line=dict(dash='solid')
                    ),
                    row=1, col=1
                )
        
        # PSNR plot
        if quality_data["psnr_data"]:
            psnr_df = pd.DataFrame(quality_data["psnr_data"])
            for encoder in psnr_df["encoder"].unique():
                encoder_data = psnr_df[psnr_df["encoder"] == encoder]
                fig.add_trace(
                    go.Scatter(
                        x=encoder_data["bitrate"],
                        y=encoder_data["psnr"],
                        mode='markers+lines',
                        name=f'PSNR - {encoder}',
                        line=dict(dash='dash')
                    ),
                    row=1, col=2
                )
        
        # SSIM plot
        if quality_data["ssim_data"]:
            ssim_df = pd.DataFrame(quality_data["ssim_data"])
            for encoder in ssim_df["encoder"].unique():
                encoder_data = ssim_df[ssim_df["encoder"] == encoder]
                fig.add_trace(
                    go.Scatter(
                        x=encoder_data["bitrate"],
                        y=encoder_data["ssim"],
                        mode='markers+lines',
                        name=f'SSIM - {encoder}',
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
            height=800,
            showlegend=True
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
        
        # Create comprehensive dashboard with multiple tabs
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        from plotly.offline import plot
        
        # Create main dashboard
        dashboard_html = self._create_dashboard_html(test_results, quality_data, quality_results)
        
        with open(report_file, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Interactive report saved to: {report_file}")
        return str(report_file)
    
    def _create_dashboard_html(self, test_results: List[TestResult], 
                             quality_data: Dict[str, Any], 
                             quality_results: List[Dict[str, Any]] = None) -> str:
        """Create comprehensive dashboard HTML"""
        from plotly.offline import plot
        
        # Generate individual plot figures
        quality_fig = self.create_quality_plots(quality_data)
        comparison_fig = self.create_comparison_plots(quality_data)
        summary_fig = self._create_summary_plots(test_results)
        error_fig = self._create_error_analysis_plots(test_results)
        
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
                {comparison_html}
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
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def create_comparison_plots(self, quality_data: Dict[str, Any]) -> go.Figure:
        """Create comparison plots for different encoders/devices"""
        if not PLOTLY_AVAILABLE:
            return None
            
        # Create subplots for encoder and device comparisons
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Encoder Comparison - VMAF", "Device Comparison - PSNR",
                          "Source Complexity Analysis", "Bitrate Efficiency"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Encoder comparison
        if quality_data["vmaf_data"]:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
            for encoder in vmaf_df["encoder"].unique():
                encoder_data = vmaf_df[vmaf_df["encoder"] == encoder]
                fig.add_trace(
                    go.Scatter(
                        x=encoder_data["bitrate"],
                        y=encoder_data["vmaf"],
                        mode='markers+lines',
                        name=f'{encoder}',
                        line=dict(width=3)
                    ),
                    row=1, col=1
                )
        
        # Device comparison
        if quality_data["psnr_data"]:
            psnr_df = pd.DataFrame(quality_data["psnr_data"])
            for device in psnr_df["device"].unique():
                device_data = psnr_df[psnr_df["device"] == device]
                fig.add_trace(
                    go.Scatter(
                        x=device_data["bitrate"],
                        y=device_data["psnr"],
                        mode='markers+lines',
                        name=f'Device {device}',
                        line=dict(width=3)
                    ),
                    row=1, col=2
                )
        
        # Source complexity analysis
        if quality_data["vmaf_data"]:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
            source_complexity = vmaf_df.groupby("source").agg({
                "vmaf": "mean",
                "bitrate": "mean"
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=source_complexity["bitrate"],
                    y=source_complexity["vmaf"],
                    mode='markers',
                    text=source_complexity["source"],
                    name="Source Complexity",
                    marker=dict(size=10, color='red')
                ),
                row=2, col=1
            )
        
        # Bitrate efficiency (VMAF per bitrate)
        if quality_data["vmaf_data"]:
            vmaf_df = pd.DataFrame(quality_data["vmaf_data"])
            vmaf_df["efficiency"] = vmaf_df["vmaf"] / vmaf_df["bitrate"]
            
            for encoder in vmaf_df["encoder"].unique():
                encoder_data = vmaf_df[vmaf_df["encoder"] == encoder]
                fig.add_trace(
                    go.Scatter(
                        x=encoder_data["bitrate"],
                        y=encoder_data["efficiency"],
                        mode='markers+lines',
                        name=f'Efficiency - {encoder}',
                        line=dict(dash='dash')
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Codec Performance Comparison",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Bitrate (kbps)", row=1, col=1)
        fig.update_yaxes(title_text="VMAF", row=1, col=1)
        fig.update_xaxes(title_text="Bitrate (kbps)", row=1, col=2)
        fig.update_yaxes(title_text="PSNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text="Bitrate (kbps)", row=2, col=1)
        fig.update_yaxes(title_text="VMAF", row=2, col=1)
        fig.update_xaxes(title_text="Bitrate (kbps)", row=2, col=2)
        fig.update_yaxes(title_text="VMAF/Bitrate", row=2, col=2)
        
        return fig
    
    def _create_summary_plots(self, test_results: List[TestResult]) -> go.Figure:
        """Create summary plots for test results"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Test Success Rate", "Encoder Performance", 
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
        
        # Encoder Performance
        encoder_stats = {}
        for result in test_results:
            encoder = result.encoder
            if encoder not in encoder_stats:
                encoder_stats[encoder] = {"success": 0, "total": 0}
            encoder_stats[encoder]["total"] += 1
            if result.success:
                encoder_stats[encoder]["success"] += 1
        
        encoders = list(encoder_stats.keys())
        success_rates = [encoder_stats[enc]["success"] / encoder_stats[enc]["total"] * 100 
                        for enc in encoders]
        
        fig.add_trace(
            go.Bar(
                x=encoders,
                y=success_rates,
                name="Success Rate %",
                marker_color='lightblue'
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
        
        # Device Performance
        device_stats = {}
        for result in test_results:
            device = result.device_serial
            if device not in device_stats:
                device_stats[device] = {"success": 0, "total": 0}
            device_stats[device]["total"] += 1
            if result.success:
                device_stats[device]["success"] += 1
        
        devices = list(device_stats.keys())
        device_success_rates = [device_stats[dev]["success"] / device_stats[dev]["total"] * 100 
                               for dev in devices]
        
        fig.add_trace(
            go.Bar(
                x=devices,
                y=device_success_rates,
                name="Device Success Rate %",
                marker_color='orange'
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