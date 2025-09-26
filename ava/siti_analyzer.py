#!/usr/bin/env python3
"""
Unified SI/TI Analysis Module
Based on the working implementation from encapp
"""

import os
import subprocess
import tempfile
import logging
from typing import Tuple, Optional
from pathlib import Path


class SITIAnalyzer:
    """Unified SI/TI analyzer using the working encapp implementation"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger('siti_analyzer')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    def analyze_si_ti(self, video_file: str) -> Tuple[float, float]:
        """
        Analyze SI/TI values using the working encapp implementation
        
        Args:
            video_file: Path to the video file to analyze
            
        Returns:
            Tuple of (SI_average, TI_average) values
        """
        if not os.path.exists(video_file):
            self.logger.error(f"Video file not found: {video_file}")
            return 0.0, 0.0
        
        try:
            # Use the working encapp approach with temporary file
            with tempfile.NamedTemporaryFile(suffix=".txt", prefix="encapp.siti.", delete=False) as tfo:
                tf = tfo.name
            
            try:
                # Build FFmpeg command using the working encapp approach
                shell_cmd = (
                    f"ffmpeg -i '{video_file}' "
                    "-filter_complex "
                    '"siti=print_summary=1" '
                    f"-f null - &> {tf}"
                )
                
                if self.debug:
                    self.logger.debug(f"Running SI/TI analysis: {shell_cmd}")
                
                # Run FFmpeg command
                result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True)
                
                # Parse the output file
                si_avg, ti_avg = self._parse_siti_file(tf)
                
                return si_avg, ti_avg
                
            finally:
                # Clean up temporary file
                if os.path.exists(tf):
                    os.unlink(tf)
                    
        except Exception as e:
            self.logger.error(f"Error analyzing SI/TI for {video_file}: {e}")
            return 0.0, 0.0
    
    def _parse_siti_file(self, file_path: str) -> Tuple[float, float]:
        """
        Parse SI/TI values from FFmpeg output file
        Based on the working encapp implementation
        """
        data = {}
        spatial = True
        
        try:
            with open(file_path, "r") as fd:
                lines = fd.readlines()
                
                for line in lines:
                    line = line.lower()
                    
                    if "spatial" in line:
                        spatial = True
                        continue
                    if "temporal" in line:
                        spatial = False
                        continue
                    
                    try:
                        key, val = line.split(": ")
                    except Exception as ve:
                        if self.debug:
                            self.logger.debug(f"SITI parse Val error: {ve} for {line}")
                        continue
                    
                    val = val.strip()
                    
                    if "average" in key:
                        if spatial:
                            data["si_avg"] = val
                        else:
                            data["ti_avg"] = val
                    if "max" in key:
                        if spatial:
                            data["si_max"] = val
                        else:
                            data["ti_max"] = val
                    if "min" in key:
                        if spatial:
                            data["si_min"] = val
                        else:
                            data["ti_min"] = val
            
            # Extract average values
            si_avg = float(data.get("si_avg", 0.0))
            ti_avg = float(data.get("ti_avg", 0.0))
            
            if self.debug:
                self.logger.debug(f"Parsed SI/TI: SI={si_avg}, TI={ti_avg}")
                self.logger.debug(f"Full data: {data}")
            
            return si_avg, ti_avg
            
        except Exception as e:
            self.logger.error(f"Error parsing SI/TI file {file_path}: {e}")
            return 0.0, 0.0


def analyze_si_ti_values(video_file: str, debug: bool = False) -> Tuple[float, float]:
    """
    Convenience function to analyze SI/TI values
    
    Args:
        video_file: Path to the video file to analyze
        debug: Enable debug logging
        
    Returns:
        Tuple of (SI_average, TI_average) values
    """
    analyzer = SITIAnalyzer(debug=debug)
    return analyzer.analyze_si_ti(video_file)


if __name__ == "__main__":
    # Test the analyzer
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python siti_analyzer.py <video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    si, ti = analyze_si_ti_values(video_file, debug=True)
    print(f"SI: {si:.2f}, TI: {ti:.2f}")
