#!/usr/bin/env python3

"""
AVA Video Source Generation

This module generates test video sources for codec testing following the guidelines
from ~/code/ava_gen/ava/ava_sources.py. It creates videos with specific characteristics
for testing different codec features and behaviors.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging


class MotionType(Enum):
    """Types of motion patterns for test videos"""
    STATIC = "static"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    LATERAL = "lateral"
    ROTATIONAL = "rotational"
    ZOOM = "zoom"
    COMPLEX = "complex"


class ComplexityProfile(Enum):
    """Complexity profiles for test videos"""
    LOW_SPATIAL_LOW_TEMPORAL = "ls.lt"
    LOW_SPATIAL_MID_TEMPORAL = "ls.mt"
    LOW_SPATIAL_HIGH_TEMPORAL = "ls.ht"
    MID_SPATIAL_LOW_TEMPORAL = "ms.lt"
    MID_SPATIAL_MID_TEMPORAL = "ms.mt"
    MID_SPATIAL_HIGH_TEMPORAL = "ms.ht"
    HIGH_SPATIAL_LOW_TEMPORAL = "hs.lt"
    HIGH_SPATIAL_MID_TEMPORAL = "hs.mt"
    HIGH_SPATIAL_HIGH_TEMPORAL = "hs.ht"


@dataclass
class VideoSpec:
    """Specification for a test video"""
    width: int
    height: int
    framerate: int
    duration: float
    motion_type: MotionType
    complexity_profile: ComplexityProfile
    color_space: str = "bt709"
    pixel_format: str = "yuv420p"
    bit_depth: int = 8




class VideoSourceGenerator:
    """Generates test video sources for codec testing"""
    
    def __init__(self, output_dir: str = "sources", debug: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debug information"""
        logger = logging.getLogger("ava_sources")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def generate_test_pattern(self, spec: VideoSpec) -> str:
        """Generate a test pattern video using FFmpeg"""
        filename = f"{spec.complexity_profile.value}_{spec.width}x{spec.height}_{spec.framerate}fps_{spec.duration}s.mp4"
        output_path = self.output_dir / filename
        
        if output_path.exists():
            self.logger.info(f"Source already exists: {filename}")
            return str(output_path)
        
        # Create FFmpeg command based on motion type and complexity
        cmd = self._build_ffmpeg_command(spec, output_path)
        
        self.logger.info(f"Generating source: {filename}")
        self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Successfully generated: {filename}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to generate {filename}: {e.stderr}")
            raise
    
    def _build_ffmpeg_command(self, spec: VideoSpec, output_path: Path) -> List[str]:
        """Build FFmpeg command for generating test video"""
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", self._get_input_filter(spec),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "0",  # Lossless
            "-pix_fmt", spec.pixel_format,
            "-color_range", "limited",
            "-colorspace", spec.color_space,
            "-color_trc", "bt709",
            "-color_primaries", "bt709",
            "-r", str(spec.framerate),
            "-t", str(spec.duration),
            str(output_path)
        ]
        return cmd
    
    def _get_input_filter(self, spec: VideoSpec) -> str:
        """Get FFmpeg input filter based on motion type and complexity"""
        base_size = f"{spec.width}x{spec.height}"
        
        if spec.motion_type == MotionType.STATIC:
            return f"color=c=black:size={base_size}:duration={spec.duration}"
        
        elif spec.motion_type == MotionType.HORIZONTAL:
            return self._get_horizontal_motion_filter(spec)
        
        elif spec.motion_type == MotionType.VERTICAL:
            return self._get_vertical_motion_filter(spec)
        
        elif spec.motion_type == MotionType.LATERAL:
            return self._get_lateral_motion_filter(spec)
        
        elif spec.motion_type == MotionType.ROTATIONAL:
            return self._get_rotational_motion_filter(spec)
        
        elif spec.motion_type == MotionType.ZOOM:
            return self._get_zoom_motion_filter(spec)
        
        elif spec.motion_type == MotionType.COMPLEX:
            return self._get_complex_motion_filter(spec)
        
        else:
            raise ValueError(f"Unknown motion type: {spec.motion_type}")
    
    def _get_horizontal_motion_filter(self, spec: VideoSpec) -> str:
        """Generate horizontal motion pattern"""
        speed = self._get_motion_speed(spec.complexity_profile)
        return (f"testsrc2=size={spec.width}x{spec.height}:duration={spec.duration}:"
                f"alpha=1,"
                f"drawbox=x='t*{speed}':y=0:w=100:h={spec.height}:color=white@0.8")
    
    def _get_vertical_motion_filter(self, spec: VideoSpec) -> str:
        """Generate vertical motion pattern"""
        speed = self._get_motion_speed(spec.complexity_profile)
        return (f"testsrc2=size={spec.width}x{spec.height}:duration={spec.duration}:"
                f"alpha=1,"
                f"drawbox=x=0:y='t*{speed}':w={spec.width}:h=100:color=white@0.8")
    
    def _get_lateral_motion_filter(self, spec: VideoSpec) -> str:
        """Generate lateral (diagonal) motion pattern"""
        speed = self._get_motion_speed(spec.complexity_profile)
        return (f"testsrc2=size={spec.width}x{spec.height}:duration={spec.duration}:"
                f"alpha=1,"
                f"drawbox=x='t*{speed}':y='t*{speed*0.5}':w=100:h=100:color=white@0.8")
    
    def _get_rotational_motion_filter(self, spec: VideoSpec) -> str:
        """Generate rotational motion pattern"""
        speed = self._get_motion_speed(spec.complexity_profile)
        center_x = spec.width // 2
        center_y = spec.height // 2
        return (f"testsrc2=size={spec.width}x{spec.height}:duration={spec.duration}:"
                f"alpha=1,"
                f"rotate=PI*t*{speed}:c=black:ow={spec.width}:oh={spec.height}")
    
    def _get_zoom_motion_filter(self, spec: VideoSpec) -> str:
        """Generate zoom motion pattern"""
        speed = self._get_motion_speed(spec.complexity_profile)
        return (f"testsrc2=size={spec.width}x{spec.height}:duration={spec.duration}:"
                f"alpha=1,"
                f"scale=iw*{1+speed*t}:ih*{1+speed*t}")
    
    def _get_complex_motion_filter(self, spec: VideoSpec) -> str:
        """Generate complex motion pattern combining multiple types"""
        speed = self._get_motion_speed(spec.complexity_profile)
        return (f"testsrc2=size={spec.width}x{spec.height}:duration={spec.duration}:"
                f"alpha=1,"
                f"drawbox=x='t*{speed}':y='sin(t*{speed})*{spec.height//4}+{spec.height//2}':"
                f"w=100:h=100:color=white@0.8,"
                f"rotate=PI*t*{speed*0.1}:c=black:ow={spec.width}:oh={spec.height}")
    
    def _get_motion_speed(self, profile: ComplexityProfile) -> float:
        """Get motion speed based on complexity profile"""
        speed_map = {
            ComplexityProfile.LOW_SPATIAL_LOW_TEMPORAL: 10.0,
            ComplexityProfile.LOW_SPATIAL_MID_TEMPORAL: 20.0,
            ComplexityProfile.LOW_SPATIAL_HIGH_TEMPORAL: 30.0,
            ComplexityProfile.MID_SPATIAL_LOW_TEMPORAL: 15.0,
            ComplexityProfile.MID_SPATIAL_MID_TEMPORAL: 25.0,
            ComplexityProfile.MID_SPATIAL_HIGH_TEMPORAL: 35.0,
            ComplexityProfile.HIGH_SPATIAL_LOW_TEMPORAL: 20.0,
            ComplexityProfile.HIGH_SPATIAL_MID_TEMPORAL: 30.0,
            ComplexityProfile.HIGH_SPATIAL_HIGH_TEMPORAL: 40.0,
        }
        return speed_map.get(profile, 20.0)
    
    def generate_framerate_test_sources(self) -> List[str]:
        """Generate sources for framerate testing"""
        sources = []
        
        # Create a complex 60fps source
        spec_60fps = VideoSpec(
            width=1920, height=1080, framerate=60, duration=10.0,
            motion_type=MotionType.COMPLEX,
            complexity_profile=ComplexityProfile.HIGH_SPATIAL_HIGH_TEMPORAL
        )
        sources.append(self.generate_test_pattern(spec_60fps))
        
        # Create a 30fps version for comparison
        spec_30fps = VideoSpec(
            width=1920, height=1080, framerate=30, duration=10.0,
            motion_type=MotionType.COMPLEX,
            complexity_profile=ComplexityProfile.HIGH_SPATIAL_HIGH_TEMPORAL
        )
        sources.append(self.generate_test_pattern(spec_30fps))
        
        return sources
    
    def generate_motion_test_sources(self) -> List[str]:
        """Generate sources for motion handling testing"""
        sources = []
        base_resolution = (1920, 1080)
        base_framerate = 30
        duration = 5.0
        
        for motion_type in [MotionType.HORIZONTAL, MotionType.LATERAL]:
            for profile in [ComplexityProfile.MID_SPATIAL_MID_TEMPORAL, 
                          ComplexityProfile.HIGH_SPATIAL_HIGH_TEMPORAL]:
                spec = VideoSpec(
                    width=base_resolution[0], height=base_resolution[1],
                    framerate=base_framerate, duration=duration,
                    motion_type=motion_type, complexity_profile=profile
                )
                sources.append(self.generate_test_pattern(spec))
        
        return sources
    
    def generate_resolution_change_sources(self) -> List[str]:
        """Generate sources for resolution change testing"""
        sources = []
        
        # Create a video that changes resolution during playback
        # This requires a more complex approach - create segments and concatenate
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create different resolution segments
            resolutions = [(1280, 720), (1920, 1080), (2560, 1440)]
            segment_files = []
            
            for i, (width, height) in enumerate(resolutions):
                spec = VideoSpec(
                    width=width, height=height, framerate=30, duration=3.0,
                    motion_type=MotionType.COMPLEX,
                    complexity_profile=ComplexityProfile.MID_SPATIAL_MID_TEMPORAL
                )
                segment_file = self.generate_test_pattern(spec)
                segment_files.append(segment_file)
            
            # Concatenate segments
            output_file = self.output_dir / "resolution_change_test.mp4"
            self._concatenate_videos(segment_files, output_file)
            sources.append(str(output_file))
            
        finally:
            # Cleanup temp files
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()
        
        return sources
    
    def _concatenate_videos(self, input_files: List[str], output_file: Path):
        """Concatenate multiple video files"""
        # Create file list for FFmpeg
        file_list = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for file_path in input_files:
            file_list.write(f"file '{file_path}'\n")
        file_list.close()
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", file_list.name,
            "-c", "copy",
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True)
        finally:
            os.unlink(file_list.name)
    
    def generate_standard_source(self, profile: ComplexityProfile, duration: float = 10.0, 
                                output_dir: Optional[str] = None) -> Optional[str]:
        """Generate a single standard test source with specified profile and duration"""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use 1280x720 as default resolution for standard sources
        spec = VideoSpec(
            width=1280, height=720, framerate=30,
            duration=duration, motion_type=MotionType.COMPLEX,
            complexity_profile=profile
        )
        
        try:
            return self.generate_test_pattern(spec)
        except Exception as e:
            self.logger.error(f"Failed to generate standard source for {profile}: {e}")
            return None

    def generate_all_standard_sources(self, duration: float = 10.0) -> List[str]:
        """Generate all standard test sources with configurable duration"""
        sources = []
        
        # Standard resolutions and framerates
        resolutions = [(1280, 720), (1920, 1080), (2560, 1440)]
        framerates = [30, 60]
        
        for width, height in resolutions:
            for framerate in framerates:
                for profile in ComplexityProfile:
                    spec = VideoSpec(
                        width=width, height=height, framerate=framerate,
                        duration=duration, motion_type=MotionType.COMPLEX,
                        complexity_profile=profile
                    )
                    sources.append(self.generate_test_pattern(spec))
        
        return sources
    
    def get_sources_for_test(self, test_name: str, duration: float = 10.0) -> List[str]:
        """Get appropriate sources for a specific test"""
        if "framerate" in test_name.lower():
            return self.generate_framerate_test_sources()
        elif "motion" in test_name.lower():
            return self.generate_motion_test_sources()
        elif "resolution" in test_name.lower():
            return self.generate_resolution_change_sources()
        else:
            # Default to standard sources
            return self.generate_all_standard_sources(duration)




if __name__ == "__main__":
    # Example usage
    generator = VideoSourceGenerator(debug=True)
    
    print("Generating test sources...")
    sources = generator.generate_all_standard_sources()
    print(f"Generated {len(sources)} test sources")
    
    # Generate specific test sources
    framerate_sources = generator.generate_framerate_test_sources()
    print(f"Generated {len(framerate_sources)} framerate test sources")
    
    motion_sources = generator.generate_motion_test_sources()
    print(f"Generated {len(motion_sources)} motion test sources")
