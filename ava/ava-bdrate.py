#!/usr/bin/env python3

"""
AVA BD-Rate Calculator

Command-line tool for calculating Bjøntegaard-Delta (BD-Rate) between video codecs
using AVA test results.

Usage:
    python ava-bdrate.py --csv results.csv --codec1 h264 --codec2 h265
    python ava-bdrate.py --workdir workdir/ --output bd_rate_report.txt
    python ava-bdrate.py --csv results.csv --all-codecs --metrics psnr,ssim,vmaf
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from typing import List, Optional

# Add the ava module to the path
sys.path.insert(0, str(Path(__file__).parent))

from bd_rate import BDRateCalculator, BDRateResult, format_bd_rate_result
from bd_rate_utils import AVABDRateAnalyzer, find_quality_csv_files, analyze_workdir_bd_rate

def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def compare_two_codecs(csv_path: str, codec1: str, codec2: str, 
                      quality_metric: str = 'psnr') -> Optional[BDRateResult]:
    """Compare two specific codecs"""
    analyzer = AVABDRateAnalyzer()
    return analyzer.calculate_bd_rate_from_csv(csv_path, codec1, codec2, quality_metric)

def compare_all_codecs(csv_path: str, quality_metrics: List[str]) -> dict:
    """Compare all codecs in the CSV file"""
    analyzer = AVABDRateAnalyzer()
    results = {}
    
    for metric in quality_metrics:
        results[metric] = analyzer.compare_all_codecs_from_csv(csv_path, metric)
    
    return results

def analyze_workdir(workdir: str, output_dir: Optional[str] = None, 
                   quality_metrics: List[str] = None) -> dict:
    """Analyze all quality CSV files in work directory"""
    if quality_metrics is None:
        quality_metrics = ['psnr', 'ssim', 'vmaf']
    
    return analyze_workdir_bd_rate(workdir, output_dir, quality_metrics)

def print_bd_rate_result(result: BDRateResult, codec1: str, codec2: str):
    """Print BD-Rate result in a formatted way"""
    print(f"\n{'='*60}")
    print(f"BD-Rate Comparison: {codec1} vs {codec2}")
    print(f"{'='*60}")
    print(format_bd_rate_result(result))
    
    # Interpretation
    if result.bd_rate > 0:
        print(f"\nInterpretation: {result.codec2} requires {abs(result.bd_rate):.2f}% more bitrate")
        print(f"than {result.codec1} for the same quality level.")
    else:
        print(f"\nInterpretation: {result.codec2} requires {abs(result.bd_rate):.2f}% less bitrate")
        print(f"than {result.codec1} for the same quality level.")

def print_all_comparisons(results: dict, quality_metric: str):
    """Print all codec comparisons for a quality metric"""
    if not results:
        print(f"No BD-Rate results available for {quality_metric}")
        return
    
    print(f"\n{'='*80}")
    print(f"BD-Rate Analysis - {quality_metric.upper()}")
    print(f"{'='*80}")
    
    for pair_name, result in results.items():
        codec1, codec2 = pair_name.split('_vs_')
        print_bd_rate_result(result, codec1, codec2)

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="Calculate Bjøntegaard-Delta (BD-Rate) between video codecs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific codecs
  python ava-bdrate.py --csv results.csv --codec1 h264 --codec2 h265
  
  # Compare all codecs with multiple metrics
  python ava-bdrate.py --csv results.csv --all-codecs --metrics psnr,ssim,vmaf
  
  # Analyze entire work directory
  python ava-bdrate.py --workdir workdir/ --output bd_rate_report.txt
  
  # Generate detailed report with all metrics
  python ava-bdrate.py --workdir workdir/ --all-metrics --output detailed_report.txt
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str, help='Path to quality CSV file')
    input_group.add_argument('--workdir', type=str, help='Path to work directory containing quality CSV files')
    
    # Codec comparison options
    codec_group = parser.add_mutually_exclusive_group(required=True)
    codec_group.add_argument('--codec1', type=str, help='First codec name for comparison')
    codec_group.add_argument('--all-codecs', action='store_true', help='Compare all codecs in the data')
    
    parser.add_argument('--codec2', type=str, help='Second codec name for comparison (required with --codec1)')
    
    # Quality metrics
    parser.add_argument('--metrics', type=str, default='psnr', 
                       help='Quality metrics to use (comma-separated): psnr,ssim,vmaf')
    parser.add_argument('--all-metrics', action='store_true', 
                       help='Use all available quality metrics')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for BD-Rate report')
    parser.add_argument('--format', type=str, choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    
    # Other options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--interpolation', type=str, choices=['linear', 'quadratic', 'cubic'], 
                       default='cubic', help='Interpolation method for RD curves')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    # Parse quality metrics
    if args.all_metrics:
        quality_metrics = ['psnr', 'ssim', 'vmaf']
    else:
        quality_metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Validate arguments
    if args.codec1 and not args.codec2:
        parser.error("--codec2 is required when using --codec1")
    
    if args.csv and args.all_codecs and not args.codec1:
        # This is valid - compare all codecs in CSV
        pass
    elif args.csv and args.codec1 and args.codec2:
        # This is valid - compare two specific codecs
        pass
    elif args.workdir:
        # This is valid - analyze work directory
        pass
    else:
        parser.error("Invalid combination of arguments")
    
    try:
        if args.workdir:
            # Analyze work directory
            logger.info(f"Analyzing work directory: {args.workdir}")
            results = analyze_workdir(args.workdir, args.output, quality_metrics)
            
            if args.output:
                logger.info(f"BD-Rate analysis saved to: {args.output}")
            else:
                # Print summary
                for file_name, file_results in results.items():
                    print(f"\nFile: {file_name}")
                    print("-" * 40)
                    for metric, metric_results in file_results.items():
                        if metric_results:
                            print_all_comparisons(metric_results, metric)
        
        elif args.csv and args.all_codecs:
            # Compare all codecs in CSV
            logger.info(f"Comparing all codecs in: {args.csv}")
            results = compare_all_codecs(args.csv, quality_metrics)
            
            for metric, metric_results in results.items():
                print_all_comparisons(metric_results, metric)
            
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    for metric, metric_results in results.items():
                        f.write(f"\n{'='*80}\n")
                        f.write(f"BD-Rate Analysis - {metric.upper()}\n")
                        f.write(f"{'='*80}\n")
                        for pair_name, result in metric_results.items():
                            codec1, codec2 = pair_name.split('_vs_')
                            f.write(f"\nComparison: {codec1} vs {codec2}\n")
                            f.write(format_bd_rate_result(result))
                            f.write("\n")
                logger.info(f"BD-Rate report saved to: {args.output}")
        
        elif args.csv and args.codec1 and args.codec2:
            # Compare two specific codecs
            logger.info(f"Comparing {args.codec1} vs {args.codec2} in: {args.csv}")
            
            for metric in quality_metrics:
                result = compare_two_codecs(args.csv, args.codec1, args.codec2, metric)
                
                if result:
                    print_bd_rate_result(result, args.codec1, args.codec2)
                else:
                    logger.warning(f"No BD-Rate result for {metric}")
            
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    for metric in quality_metrics:
                        result = compare_two_codecs(args.csv, args.codec1, args.codec2, metric)
                        if result:
                            f.write(f"\n{'='*60}\n")
                            f.write(f"BD-Rate Comparison: {args.codec1} vs {args.codec2} ({metric.upper()})\n")
                            f.write(f"{'='*60}\n")
                            f.write(format_bd_rate_result(result))
                            f.write("\n")
                logger.info(f"BD-Rate report saved to: {args.output}")
        
        else:
            parser.error("Invalid argument combination")
    
    except Exception as e:
        logger.error(f"BD-Rate calculation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
