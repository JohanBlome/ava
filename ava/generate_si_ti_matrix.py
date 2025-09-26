#!/usr/bin/env python3

"""
Analyze SI/TI Matrix from Existing Video Sources

This script analyzes existing video files to create a comprehensive matrix
of spatial and temporal complexity for codec testing.

Examples:
    # Analyze existing video sources
    python3 -m ava.generate_si_ti_matrix --sources-dir ./my_sources
    
    # Analyze with custom output directory
    python3 -m ava.generate_si_ti_matrix --sources-dir ./my_sources --output-dir ./analysis_results
    
    # Analyze with debug output
    python3 -m ava.generate_si_ti_matrix --sources-dir ./my_sources --debug
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import glob

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .siti_analyzer import SITIAnalyzer


def setup_logging(debug: bool = False, output_dir: str = "analysis") -> logging.Logger:
    """Setup logging to file instead of console"""
    logger = logging.getLogger('si_ti_matrix_analyzer')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    log_file = Path(output_dir) / "si_ti_analysis.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def find_video_files(sources_dir: str) -> List[Path]:
    """Find all video files in the specified directory"""
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v']
    video_files = []
    
    sources_path = Path(sources_dir)
    if not sources_path.exists():
        raise FileNotFoundError(f"Sources directory not found: {sources_dir}")
    
    for ext in video_extensions:
        pattern = f"**/*{ext}"
        video_files.extend(sources_path.glob(pattern))
    
    return sorted(video_files)


def analyze_video_file(analyzer: SITIAnalyzer, video_path: Path) -> Dict[str, Any]:
    """Analyze a single video file and return SI/TI data"""
    print(f"📹 Analyzing: {video_path.name}")
    
    try:
        si, ti = analyzer.analyze_si_ti(str(video_path))
        
        # Get file size
        file_size = video_path.stat().st_size
        
        return {
            'filename': video_path.name,
            'path': str(video_path),
            'si': si,
            'ti': ti,
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        print(f"❌ Error analyzing {video_path.name}: {e}")
        return {
            'filename': video_path.name,
            'path': str(video_path),
            'si': 0.0,
            'ti': 0.0,
            'file_size': 0,
            'file_size_mb': 0.0,
            'error': str(e)
        }


def analyze_existing_sources(sources_dir: str, output_dir: str = "analysis", debug: bool = False) -> List[Dict[str, Any]]:
    """Analyze existing video sources and generate SI/TI matrix"""
    logger = setup_logging(debug, output_dir)
    
    print(f"🔍 Analyzing existing sources in {sources_dir}...")
    logger.info(f"Analyzing existing sources in {sources_dir}...")
    
    # Find video files
    video_files = find_video_files(sources_dir)
    
    if not video_files:
        print("❌ No video files found in the specified directory")
        logger.error("No video files found in the specified directory")
        return []
    
    print(f"📁 Found {len(video_files)} video files")
    logger.info(f"Found {len(video_files)} video files")
    
    # Initialize analyzer
    analyzer = SITIAnalyzer(debug=debug)
    
    # Analyze all videos
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"📊 Progress: {i}/{len(video_files)}", end='\r')
        result = analyze_video_file(analyzer, video_path)
        results.append(result)
    
    print()  # New line after progress
    
    # Generate analysis summary
    successful_results = [r for r in results if r['si'] > 0 or r['ti'] > 0]
    failed_results = [r for r in results if r['si'] == 0 and r['ti'] == 0]
    
    if successful_results:
        si_values = [r['si'] for r in successful_results]
        ti_values = [r['ti'] for r in successful_results]
        file_sizes = [r['file_size'] for r in successful_results]
        
        print(f"📊 Analysis Summary:")
        print(f"   ✅ Successful: {len(successful_results)}")
        print(f"   ❌ Failed: {len(failed_results)}")
        print(f"   📈 SI range: {min(si_values):.1f} - {max(si_values):.1f}")
        print(f"   📈 TI range: {min(ti_values):.1f} - {max(ti_values):.1f}")
        print(f"   📈 SI mean: {sum(si_values)/len(si_values):.1f}")
        print(f"   📈 TI mean: {sum(ti_values)/len(ti_values):.1f}")
        print(f"   💾 File size range: {min(file_sizes):,} - {max(file_sizes):,} bytes")
        
        logger.info(f"Analysis Summary:")
        logger.info(f"SI range: {min(si_values):.1f} - {max(si_values):.1f}")
        logger.info(f"TI range: {min(ti_values):.1f} - {max(ti_values):.1f}")
        logger.info(f"SI mean: {sum(si_values)/len(si_values):.1f}")
        logger.info(f"TI mean: {sum(ti_values)/len(ti_values):.1f}")
    else:
        print("❌ No videos were successfully analyzed")
        logger.error("No videos were successfully analyzed")
    
    # Save results to JSON
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "si_ti_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    logger.info(f"Results saved to: {results_file}")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze existing video sources for SI/TI matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze existing video sources
    python3 -m ava.generate_si_ti_matrix --sources-dir ./my_sources
    
    # Analyze with custom output directory
    python3 -m ava.generate_si_ti_matrix --sources-dir ./my_sources --output-dir ./analysis_results
    
    # Analyze with debug output
    python3 -m ava.generate_si_ti_matrix --sources-dir ./my_sources --debug
        """
    )
    
    parser.add_argument(
        '--sources-dir', 
        required=True,
        help='Directory containing video files to analyze'
    )
    
    parser.add_argument(
        '--output-dir',
        default='analysis',
        help='Output directory for analysis results (default: analysis)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_existing_sources(args.sources_dir, args.output_dir, args.debug)
        
        if results:
            print(f"\n🎉 Analysis complete! Found {len(results)} video files.")
        else:
            print("\n❌ No video files found or analysis failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()