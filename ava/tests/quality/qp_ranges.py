"""
Quality Tests - Quantization Parameter Range Tests

Tests for QP range encoding quality including:
- Different quality tiers with QP ranges
- Source complexity variations
- QP range optimization (27-38 for all I, P, B frames)
"""

from ava.ava import get_resolution_from_file
from ava.ava_tests import BaseTest, TestConfig, TestResult, _build_single_test_pb
from pathlib import Path
import logging
from typing import List, Dict, Any

from ava.tests.quality.bitratemode import BitrateModeTest

logger = logging.getLogger(__name__)
class QPRangeTest(BitrateModeTest):
    """Quantization Parameter range quality tests."""
    
    mode = "vbr"
    bitrates = "8M,16M,32M"

    # TODO: clean up this, we do not have differetn tiers
    # QP ranges for different quality tiers - meaningful differences
    qp_ranges = {
        "high_quality": {
            "qp_i_min": 18, "qp_i_max": 25, 
            "qp_p_min": 20, "qp_p_max": 28, 
            "qp_b_min": 22, "qp_b_max": 30,
            "description": "High quality - lower QP values"
        },
        "medium_quality": {
            "qp_i_min": 22, "qp_i_max": 30, 
            "qp_p_min": 24, "qp_p_max": 32, 
            "qp_b_min": 26, "qp_b_max": 34,
            "description": "Medium quality - balanced QP values"
        },
        "low_quality": {
            "qp_i_min": 28, "qp_i_max": 38, 
            "qp_p_min": 30, "qp_p_max": 40, 
            "qp_b_min": 32, "qp_b_max": 42,
            "description": "Low quality - higher QP values"
        }
    }

    def __init__(self):
        super().__init__("qp_range", "Quantization Parameter Range Quality Tests")
    
    def generate_sources(self, config: TestConfig) -> List[Path]:
        """
        Generate test sources for QP range tests if they don't exist.
        
        Creates a subset of sources for QP range testing using real resolution settings.
        """
        from ava.ava_sources import make_lossless_hevc_source
        
        sources_dir = config.workdir.parent / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        
        # QP range tests use a smaller subset of sources for efficiency
        test_resolutions = self.common_resolutions
        test_framerates = [30]  # Only 30fps for QP tests
        
        # Use full complexity matrix for QP range tests
        test_profiles = [
            "ls.lt",  # low_spatial_low_temporal
            "ls.mt",  # low_spatial_mid_temporal
            "ls.ht",  # low_spatial_high_temporal
            "ms.lt",  # mid_spatial_low_temporal
            "ms.mt",  # mid_spatial_mid_temporal
            "ms.ht",  # mid_spatial_high_temporal
            "hs.lt",  # high_spatial_low_temporal
            "hs.mt",  # high_spatial_mid_temporal
            "hs.ht"   # high_spatial_high_temporal
        ]
        generated_sources = []
        
        # Calculate total number of files to generate
        total_files = len(test_resolutions) * len(test_framerates) * len(test_profiles)
        files_to_generate = []
        existing_files = []
        
        # First pass: check which files exist and which need generation
        for res in test_resolutions:
            for fps in test_framerates:
                for profile in test_profiles:
                    mp4_file = sources_dir / f"{profile}_{res}_{fps}fps_5s.mp4"
                    
                    if not mp4_file.exists():
                        files_to_generate.append((profile, mp4_file, res, fps))
                    else:
                        existing_files.append(mp4_file)
                    generated_sources.append(mp4_file)
        
        # Show progress summary only in debug mode
        if config.debug:
            print(f"📹 QP Range Source Generation:")
            print(f"   Total files: {total_files}")
            print(f"   Existing: {len(existing_files)}")
            print(f"   To generate: {len(files_to_generate)}")
        
        if files_to_generate:
            if config.debug:
                print(f"   Generating sources...")
            
            # Generate missing files with progress reporting
            for i, (profile, mp4_file, res, fps) in enumerate(files_to_generate, 1):
                if config.debug:
                    print(f"   [{i}/{len(files_to_generate)}] Generating {mp4_file.name}...")
                try:
                    # Use a simple test content profile
                    make_lossless_hevc_source(profile, mp4_file, res, fps, 5.0)
                    if config.debug:
                        print(f"      ✅ Generated: {mp4_file.name}")
                except Exception as e:
                    if config.debug:
                        print(f"      ❌ Failed: {mp4_file.name} - {e}")
                    raise
        else:
            if config.debug:
                print(f"   ✅ All sources already exist, using existing files")
        
        if config.debug:
            print(f"📁 Source directory: {sources_dir}")
        return generated_sources
    
    def generate_tests(self, config: TestConfig) -> List[Dict[str, Any]]:
        """Generate QP range test configurations grouped by source."""
        # Use all available MP4 sources - encapp handles decoding automatically
        sources = [s for s in config.sources if s.suffix == '.mp4']
 
        # Group tests by source - each source gets one test group
        test_groups = []
        for source in sources:
            source_name = source.stem
            source_tests = []

            resolution = get_resolution_from_file(source)

            for quality_tier, qp_params in self.qp_ranges.items():
                for bitrate in self.bitrates:
                    test_id = f"QUALITY.QP_RANGE.qp.{quality_tier}.{source_name}.{self.mode}.{bitrate}kbps"
                    description = f"QP range {quality_tier}: {qp_params['description']} (I:{qp_params['qp_i_min']}-{qp_params['qp_i_max']}, P:{qp_params['qp_p_min']}-{qp_params['qp_p_max']}, B:{qp_params['qp_b_min']}-{qp_params['qp_b_max']}) on {source_name}"
                    
                    source_tests.append({
                        "short_id": test_id,
                        "elaborate_description": description,
                        "source": source,
                        "qp_params": qp_params,
                        "quality_tier": quality_tier,
                        "bitrate": bitrate
                    })
            
            # Add the source group
            test_groups.append({
                "source": source,
                "source_name": source_name,
                "tests": source_tests
            })
        
        return test_groups
    

    def execute_test(self, test_group: Dict[str, Any], config: TestConfig) -> List[TestResult]:
        """Execute a group of QP range tests for a single source."""
        results = []
        source = test_group["source"]
        source_name = test_group["source_name"]
        tests = test_group["tests"]
        
        try:
            # Generate pbtxt with multiple tests for this source
            # Use MP4 decoding as first priority (surface: true, device_decode: true)
            pbtxt_content = self._generate_grouped_pbtxt(tests, source, config, use_surface_transcoding=True)
            
            # Write pbtxt to file (one file per source)
            pbtxt_file = config.workdir / f"qp_{source_name}.pbtxt"
            pbtxt_file.write_text(pbtxt_content)
            
            # Run encapp test with timeout (first test gets 60s, others get 2x average)
            if not hasattr(self, '_execution_times'):
                self._execution_times = []
            timeout_threshold = 60.0 if not self._execution_times else sum(self._execution_times) / len(self._execution_times) * 2.0
            
            success, execution_time = self._run_encapp_test(pbtxt_file, config, timeout_threshold)
            self._execution_times.append(execution_time)
            
        except Exception as e:
            logger.error(f"Test failed for {source_name}: {e}")
            # Create failed results for all tests in this group TODO: look at this
            for test in tests:
                result = TestResult(
                    test_id=test["short_id"],
                    category="quality",
                    module="qp_ranges",
                    device_serial=config.device_serial,
                    success=False,
                    error=str(e),
                    metrics={
                        "codec": config.codec,
                        "resolution": config.resolution,
                        "fps": config.fps,
                        "bitrate": test["bitrate"],
                        "bitrate_mode": test["birate_mode"],
                        "qp_i_min": test["qp_params"]["qp_i_min"],
                        "qp_i_max": test["qp_params"]["qp_i_max"],
                        "qp_p_min": test["qp_params"]["qp_p_min"],
                        "qp_p_max": test["qp_params"]["qp_p_max"],
                        "qp_b_min": test["qp_params"]["qp_b_min"],
                        "qp_b_max": test["qp_params"]["qp_b_max"],
                        "quality_tier": test["quality_tier"]
                    },
                    description=test.get("elaborate_description")
                )
                results.append(result)
            return results
        
        # Create successful test results for all tests in this group
        for test in tests:
            result = TestResult(
                test_id=test["short_id"],
                category="quality",
                module="qp_ranges",
                device_serial=config.device_serial,
                success=True,
                error=None,
                metrics={    # hm... TODO: fix values sources
                    "codec": config.codec,
                    "resolution": config.resolution,
                    "fps": config.fps,
                    "bitrate_mode": test["bitrate_mode"],
                    "qp_i_min": test["qp_params"]["qp_i_min"],
                    "qp_i_max": test["qp_params"]["qp_i_max"],
                    "qp_p_min": test["qp_params"]["qp_p_min"],
                    "qp_p_max": test["qp_params"]["qp_p_max"],
                    "qp_b_min": test["qp_params"]["qp_b_min"],
                    "qp_b_max": test["qp_params"]["qp_b_max"],
                    "quality_tier": test["quality_tier"]
                },
                description=test.get("elaborate_description")
            )
            results.append(result)
        
        return results
    
    def _generate_grouped_pbtxt(self, tests: List[Dict], source: Path, config: TestConfig, 
                               use_surface_transcoding: bool) -> str:
        """Generate a pbtxt file with multiple tests for the same source."""
        test_blocks = []
        
        for test in tests:
            # Build QP range parameters using Parameter items in additional_config
            qp_params = test["qp_params"]
            qp_additional_config = f"""    parameter {{ key: "video-qp-i-min" type: intType value: "{qp_params['qp_i_min']}" }}
    parameter {{ key: "video-qp-i-max" type: intType value: "{qp_params['qp_i_max']}" }}
    parameter {{ key: "video-qp-p-min" type: intType value: "{qp_params['qp_p_min']}" }}
    parameter {{ key: "video-qp-p-max" type: intType value: "{qp_params['qp_p_max']}" }}
    parameter {{ key: "video-qp-b-min" type: intType value: "{qp_params['qp_b_min']}" }}
    parameter {{ key: "video-qp-b-max" type: intType value: "{qp_params['qp_b_max']}" }}"""
            
            # Generate test block with QP parameters in additional_config
            test_block = _build_single_test_pb(
                name=test["short_id"],
                outfile=f"{test['short_id']}.qp{qp_params['qp_i_min']}-{qp_params['qp_i_max']}",
                source=source,
                codec=config.codec,
                fps=config.fps,
                i_frame_interval=30,
                bitrate_mode=self.mode,  # QP tests use VBR mode
                bitrate=None,  # QP mode doesn't use bitrate
                use_surface_transcoding=use_surface_transcoding,
                use_lossless_hevc=use_surface_transcoding,
                description=test["elaborate_description"],
                additional_config=qp_additional_config
            )
            
            test_blocks.append(test_block)
        
        # Combine all test blocks into a single pbtxt
        return "\n\n".join(test_blocks)

# Test is registered in test_runner.py