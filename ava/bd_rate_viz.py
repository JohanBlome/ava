#!/usr/bin/env python3

"""
BD-Rate Visualization Module

This module provides visualization functions for BD-Rate analysis results,
including rate-distortion curves, BD-Rate bar charts, and comprehensive
comparison plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    from .bd_rate import BDRateResult, BDRateCalculator
except ImportError:
    from bd_rate import BDRateResult, BDRateCalculator

class BDRateVisualizer:
    """Visualization tools for BD-Rate analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculator = BDRateCalculator()
    
    def create_rd_curve_plot(self, rd_data: Dict[str, pd.DataFrame], 
                           quality_metric: str = 'psnr',
                           title: str = None) -> go.Figure:
        """
        Create rate-distortion curve plot
        
        Args:
            rd_data: Dictionary mapping codec names to RD curve DataFrames
            quality_metric: Quality metric name
            title: Optional plot title
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f"Rate-Distortion Curves ({quality_metric.upper()})"
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        color_idx = 0
        
        # Debug logging
        self.logger.info(f"Creating RD curve plot for {quality_metric} with {len(rd_data)} codecs")
        
        for codec, data in rd_data.items():
            self.logger.info(f"Codec {codec}: {len(data)} data points")
            if data.empty:
                self.logger.warning(f"Empty data for codec {codec}")
                continue
            
            # Sort by quality for proper curve
            data_sorted = data.sort_values('quality')
            
            # Convert bitrate to kbps for better readability
            bitrates_kbps = data_sorted['bitrate'] / 1000
            qualities = data_sorted['quality']
            
            
            fig.add_trace(go.Scatter(
                x=data_sorted['bitrate'] / 1000,  # Convert to kbps
                y=data_sorted['quality'],
                mode='markers+lines',
                name=codec,
                line=dict(color=colors[color_idx % len(colors)], width=3),
                marker=dict(size=8, color=colors[color_idx % len(colors)]),
                hovertemplate=f'<b>{codec}</b><br>' +
                             f'Bitrate: %{{x:.1f}} kbps<br>' +
                             f'{quality_metric.upper()}: %{{y:.2f}}<extra></extra>'
            ))
            color_idx += 1
        
        fig.update_layout(
            title=title,
            xaxis_title="Bitrate (kbps)",
            yaxis_title=f"{quality_metric.upper()}",
            hovermode='closest',
            height=600,
            width=800,
            showlegend=True,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Set proper axis ranges based on data
        if len(rd_data) > 0:
            all_bitrates = []
            all_qualities = []
            for data in rd_data.values():
                if not data.empty:
                    all_bitrates.extend((data['bitrate'] / 1000).tolist())  # Convert to kbps
                    all_qualities.extend(data['quality'].tolist())
            
            if all_bitrates and all_qualities:
                # Calculate ranges with better padding
                x_min = min(all_bitrates)
                x_max = max(all_bitrates)
                y_min = min(all_qualities)
                y_max = max(all_qualities)
                
                # Add 10% padding to both axes
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_padding = x_range * 0.1 if x_range > 0 else x_max * 0.1
                y_padding = y_range * 0.1 if y_range > 0 else y_max * 0.1
                
                x_min_padded = max(0, x_min - x_padding)  # Don't go below 0 for bitrate
                x_max_padded = x_max + x_padding
                y_min_padded = y_min - y_padding
                y_max_padded = y_max + y_padding
                
                # Set axis ranges
                fig.update_xaxes(range=[x_min_padded, x_max_padded])
                fig.update_yaxes(range=[y_min_padded, y_max_padded])
                
                # Use linear scale for bitrate to ensure curves are visible
                fig.update_xaxes(type="linear", range=[x_min_padded, x_max_padded])
        
        return fig
    
    def create_reference_bd_rate_chart(self, bd_results: Dict[str, BDRateResult], 
                                     reference_codec: str,
                                     quality_metric: str = 'psnr',
                                     title: str = None) -> go.Figure:
        """
        Create BD-Rate bar chart with reference codec
        
        Args:
            bd_results: Dictionary mapping codec names to BD-Rate results
            reference_codec: Name of the reference codec
            quality_metric: Quality metric name
            title: Optional plot title
            
        Returns:
            Plotly figure
        """
        if not bd_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No BD-Rate data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=16, color="red")
            )
            return fig
        
        if title is None:
            title = f"BD-Rate vs {reference_codec} ({quality_metric.upper()})"
        
        # Prepare data for plotting
        codecs = list(bd_results.keys())
        bd_rates = [bd_results[codec].bd_rate for codec in codecs]
        
        # Color code: green for negative (better), red for positive (worse)
        colors = ['green' if rate < 0 else 'red' for rate in bd_rates]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=codecs,
            y=bd_rates,
            marker_color=colors,
            text=[f"{rate:.2f}%" for rate in bd_rates],
            textposition='auto',
            hovertemplate=f'<b>%{{x}}</b><br>BD-Rate vs {reference_codec}: %{{y:.2f}}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Codecs",
            yaxis_title=f"BD-Rate vs {reference_codec} (%)",
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
            showlegend=False,
            height=600,
            width=800,
            margin=dict(l=60, r=60, t=80, b=100)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_bd_rate_bar_chart(self, bd_results: Dict[str, BDRateResult], 
                               quality_metric: str = 'psnr',
                               title: str = None) -> go.Figure:
        """
        Create BD-Rate comparison bar chart
        
        Args:
            bd_results: Dictionary of BD-Rate results
            quality_metric: Quality metric name
            title: Optional plot title
            
        Returns:
            Plotly figure
        """
        if not bd_results:
            fig = go.Figure()
            fig.add_annotation(
                text="No BD-Rate data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        if title is None:
            title = f"BD-Rate Comparison ({quality_metric.upper()})"
        
        # Extract data for plotting
        codec_pairs = []
        bd_rates = []
        colors = []
        hover_text = []
        
        for pair_name, result in bd_results.items():
            codec_pairs.append(pair_name.replace('_vs_', ' vs '))
            bd_rates.append(result.bd_rate)
            
            # Color based on positive/negative BD-Rate
            if result.bd_rate > 0:
                colors.append('#ff6b6b')  # Red for higher bitrate (worse)
            else:
                colors.append('#51cf66')  # Green for lower bitrate (better)
            
            # Hover text with additional info
            hover_text.append(
                f"<b>{pair_name.replace('_vs_', ' vs ')}</b><br>" +
                f"BD-Rate: {result.bd_rate:.2f}%<br>" +
                f"Quality Range: {result.integration_range[0]:.2f} - {result.integration_range[1]:.2f}<br>" +
                f"Data Points: {result.num_points}<br>" +
                f"Interpolation: {result.interpolation_method}"
            )
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=codec_pairs,
                y=bd_rates,
                marker_color=colors,
                text=[f"{rate:.2f}%" for rate in bd_rates],
                textposition='auto',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_text
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Codec Pairs",
            yaxis_title="BD-Rate (%)",
            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=2),
            showlegend=False,
            height=600,
            width=800,
            margin=dict(l=60, r=60, t=80, b=100)  # Extra bottom margin for rotated labels
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
    
    def create_comprehensive_comparison(self, rd_data: Dict[str, pd.DataFrame], 
                                      bd_results: Dict[str, BDRateResult],
                                      quality_metric: str = 'psnr') -> go.Figure:
        """
        Create comprehensive comparison plot with RD curves and BD-Rate bars
        
        Args:
            rd_data: Dictionary mapping codec names to RD curve DataFrames
            bd_results: Dictionary of BD-Rate results
            quality_metric: Quality metric name
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"Rate-Distortion Curves ({quality_metric.upper()})",
                f"BD-Rate Comparison ({quality_metric.upper()})"
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add RD curves to first subplot
        colors = px.colors.qualitative.Set1
        color_idx = 0
        
        for codec, data in rd_data.items():
            if data.empty:
                continue
            
            data_sorted = data.sort_values('quality')
            
            fig.add_trace(go.Scatter(
                x=data_sorted['bitrate'] / 1000,  # Convert to kbps
                y=data_sorted['quality'],
                mode='markers+lines',
                name=codec,
                line=dict(color=colors[color_idx % len(colors)], width=3),
                marker=dict(size=6),
                showlegend=True,
                hovertemplate=f'<b>{codec}</b><br>' +
                             f'Bitrate: %{{x:.1f}} kbps<br>' +
                             f'{quality_metric.upper()}: %{{y:.2f}}<extra></extra>'
            ), row=1, col=1)
            color_idx += 1
        
        # Add BD-Rate bars to second subplot
        if bd_results:
            codec_pairs = []
            bd_rates = []
            colors = []
            
            for pair_name, result in bd_results.items():
                codec_pairs.append(pair_name.replace('_vs_', ' vs '))
                bd_rates.append(result.bd_rate)
                
                if result.bd_rate > 0:
                    colors.append('#ff6b6b')
                else:
                    colors.append('#51cf66')
            
            fig.add_trace(go.Bar(
                x=codec_pairs,
                y=bd_rates,
                marker_color=colors,
                text=[f"{rate:.2f}%" for rate in bd_rates],
                textposition='auto',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>BD-Rate: %{y:.2f}%<extra></extra>'
            ), row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title=f"Codec Performance Analysis - {quality_metric.upper()}",
            height=600,
            width=800,
            showlegend=True,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Bitrate (kbps)", row=1, col=1)
        fig.update_yaxes(title_text=f"{quality_metric.upper()}", row=1, col=1)
        fig.update_xaxes(title_text="Codec Pairs", row=1, col=2)
        fig.update_yaxes(title_text="BD-Rate (%)", row=1, col=2)
        
        # Rotate x-axis labels for BD-Rate plot
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        # Add zero line for BD-Rate plot
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
        
        return fig
    
    def create_quality_metric_comparison(self, bd_results_by_metric: Dict[str, Dict[str, BDRateResult]]) -> go.Figure:
        """
        Create comparison plot across multiple quality metrics
        
        Args:
            bd_results_by_metric: Dictionary mapping quality metrics to BD-Rate results
            
        Returns:
            Plotly figure
        """
        if not bd_results_by_metric:
            fig = go.Figure()
            fig.add_annotation(
                text="No BD-Rate data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Collect all unique codec pairs
        all_pairs = set()
        for metric_results in bd_results_by_metric.values():
            all_pairs.update(metric_results.keys())
        
        all_pairs = sorted(list(all_pairs))
        
        # Create subplots for each quality metric
        metrics = list(bd_results_by_metric.keys())
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=[f"{metric.upper()}" for metric in metrics],
            specs=[[{"secondary_y": False}] * len(metrics)]
        )
        
        for i, metric in enumerate(metrics):
            metric_results = bd_results_by_metric[metric]
            
            codec_pairs = []
            bd_rates = []
            colors = []
            
            for pair in all_pairs:
                if pair in metric_results:
                    result = metric_results[pair]
                    codec_pairs.append(pair.replace('_vs_', ' vs '))
                    bd_rates.append(result.bd_rate)
                    
                    if result.bd_rate > 0:
                        colors.append('#ff6b6b')
                    else:
                        colors.append('#51cf66')
                else:
                    codec_pairs.append(pair.replace('_vs_', ' vs '))
                    bd_rates.append(0)
                    colors.append('#gray')
            
            fig.add_trace(go.Bar(
                x=codec_pairs,
                y=bd_rates,
                marker_color=colors,
                text=[f"{rate:.2f}%" if rate != 0 else "N/A" for rate in bd_rates],
                textposition='auto',
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>BD-Rate: %{y:.2f}%<extra></extra>'
            ), row=1, col=i+1)
        
        fig.update_layout(
            title="BD-Rate Comparison Across Quality Metrics",
            height=600,
            width=800,
            showlegend=False,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Update axes for each subplot
        for i in range(len(metrics)):
            fig.update_xaxes(tickangle=45, row=1, col=i+1)
            fig.update_yaxes(title_text="BD-Rate (%)", row=1, col=i+1)
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=i+1)
        
        return fig
    
    def create_rd_curve_with_bd_rate_annotation(self, rd_data: Dict[str, pd.DataFrame], 
                                              bd_result: BDRateResult,
                                              quality_metric: str = 'psnr') -> go.Figure:
        """
        Create RD curve plot with BD-Rate annotation
        
        Args:
            rd_data: Dictionary mapping codec names to RD curve DataFrames
            bd_result: BD-Rate result for annotation
            quality_metric: Quality metric name
            
        Returns:
            Plotly figure
        """
        fig = self.create_rd_curve_plot(rd_data, quality_metric)
        
        # Add BD-Rate annotation
        bd_text = f"BD-Rate: {bd_result.bd_rate:.2f}%<br>"
        bd_text += f"Quality Range: {bd_result.integration_range[0]:.2f} - {bd_result.integration_range[1]:.2f}<br>"
        bd_text += f"Data Points: {bd_result.num_points}"
        
        fig.add_annotation(
            text=bd_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig

def create_bd_rate_summary_table(bd_results: Dict[str, BDRateResult]) -> go.Figure:
    """
    Create a summary table of BD-Rate results
    
    Args:
        bd_results: Dictionary of BD-Rate results
        
    Returns:
        Plotly figure with table
    """
    if not bd_results:
        fig = go.Figure()
        fig.add_annotation(
            text="No BD-Rate data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Prepare table data
    codec_pairs = []
    bd_rates = []
    quality_ranges = []
    data_points = []
    interpretations = []
    
    for pair_name, result in bd_results.items():
        codec_pairs.append(pair_name.replace('_vs_', ' vs '))
        bd_rates.append(f"{result.bd_rate:.2f}%")
        quality_ranges.append(f"{result.integration_range[0]:.2f} - {result.integration_range[1]:.2f}")
        data_points.append(str(result.num_points))
        
        if result.bd_rate > 0:
            interpretations.append(f"{result.codec2} needs {abs(result.bd_rate):.2f}% more bitrate")
        else:
            interpretations.append(f"{result.codec2} needs {abs(result.bd_rate):.2f}% less bitrate")
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Codec Pair", "BD-Rate", "Quality Range", "Data Points", "Interpretation"],
            fill_color="lightblue",
            align="left",
            font=dict(size=12, color="white")
        ),
        cells=dict(
            values=[codec_pairs, bd_rates, quality_ranges, data_points, interpretations],
            fill_color="white",
            align="left",
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title="BD-Rate Summary Table",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
