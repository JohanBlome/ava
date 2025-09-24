"""
AVA Video Codec Testing Framework

A comprehensive testing framework for video codec performance evaluation.
"""

__version__ = "2.0.0"
__author__ = "AVA Team"

# Import main components for easy access
from .ava import main, IntegratedTestRunner, TestConfig
from .ava_sources import VideoSourceGenerator, VideoSpec, MotionType, ComplexityProfile
from .ava_quality import QualityAssessment, QualityMetrics
from .ava_report import ReportGenerator, TestResult

__all__ = [
    "main",
    "IntegratedTestRunner", 
    "TestConfig",
    "VideoSourceGenerator",
    "VideoSpec",
    "MotionType", 
    "ComplexityProfile",
    "QualityAssessment",
    "QualityMetrics",
    "ReportGenerator",
    "TestResult"
]
