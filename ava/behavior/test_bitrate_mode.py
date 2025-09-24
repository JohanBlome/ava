"""
Bitrate mode tests for video encoders.
Tests VBR, CBR, CQ, and QP Range modes across different bitrate ladders.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .. import ava_common
    from ..ava_quality import QualityAssessment
    encapp = ava_common.encapp
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import ava_common
    from ava_quality import QualityAssessment
    encapp = ava_common.encapp

# Define pixelrates and bitrates in kbps
pixelrates = {1920 * 1080 * 30: 8000, 1920 * 1080 * 60: 10000, 1280 * 720 * 30: 6000}

def find_bitrate_for_video(videopath: str, encoder: str) -> str:
    """Find appropriate bitrate for a video based on its resolution and framerate"""
    try:
        videoinfo = ava_common.get_video_info(videopath)
        width = videoinfo.get("width", 1280)
        height = videoinfo.get("height", 720)
        framerate = videoinfo.get("framerate", 30)
        
        pixelrate = width * height * framerate
        
        # Find the closest higher pixelrate in our table
        higher = min([p for p in pixelrates.keys() if p >= pixelrate], default=max(pixelrates.keys()))
        ratio = pixelrate / higher
        bitrate = int(ratio * pixelrates[higher])
        return f"{bitrate}kbps"
    except Exception:
        # Fallback to a reasonable default
        return "2000kbps"

def _generate_test_sources(workdir, duration=10.0):
    """Generate test sources for bitrate ladder testing"""
    try:
        from .ava_sources import VideoSourceGenerator
        generator = VideoSourceGenerator(debug=True)
        sources = generator.generate_all_standard_sources(duration)
        return sources
    except ImportError:
        return []

def test_bitrate_mode_support(device, input_file, workdir, test_data):
    """Test bitrate mode support for the encoder."""
    print(f"Testing bitrate mode support for {device['encoder']}")
    
    if not encapp:
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    # Initialize test data
    test_data["test_name"] = "bitrate_mode_support"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        # Test different bitrate modes to check support
        bitrate_modes = [
            ("vbr", encapp.tests_definitions.Configure.BitrateMode.vbr),
            ("cbr", encapp.tests_definitions.Configure.BitrateMode.cbr),
            ("cq", encapp.tests_definitions.Configure.BitrateMode.cq),
        ]
        
        mode_support = {}
        output_files = []
        
        for mode_name, mode_enum in bitrate_modes:
            print(f"  Testing {mode_name.upper()} mode support")
            
            try:
                # Create a simple test for this mode
                testdata = ava_common.initialize_testdata()
                test_suite = encapp.tests_definitions.TestSuite()
                test = encapp.tests_definitions.Test()

                # Setup test for MP4 input with device decoding
                ava_common.setup_test_for_mp4_input(
                    test,
                    input_file,
                    device,
                    "/tmp/",  # mediastore (host path)
                )

                # Enable device decoding for MP4 transcoding
                test.input.device_decode = True

                # Create unique test ID
                test.common.id = f"{__name__}.mode_test_{mode_name}.{test.common.id}"
                test.common.description = f"Test {mode_name.upper()} mode support"
                test.common.output_filename = test.common.id

                # Configure the test
                test.input.realtime = False
                test.configure.codec = device["encoder"]
                test.configure.bitrate = "1Mbps"
                test.configure.bitrate_mode = mode_enum
                
                # Add to test suite
                test_suite.test.extend([test])
                pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
                encapp.configfile_write(test_suite, pbtxt_file)
                testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

                # Run the test
                result = encapp.run_codec_tests(
                    test_suite,
                    [],
                    "na",
                    device["serial"],
                    "/tmp/",  # mediastore (host path)
                    workdir,
                    device_workdir=device["device_workdir"],
                    ignore_results=False,
                    fast_copy=True,
                    split=False,
                    debug=0,
                )
                
                if result[0]:  # Success
                    output_files.extend(result[1])
                    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(result[1][0]).name
                    
                    # Parse results
                    with open(result[1][0], "r") as f:
                        jsonfile = json.load(f)
                    
                    mediafile = f"{workdir}/{jsonfile.get('encodedfile')}"
                    testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mediafile).name
                    
                    mode_support[mode_name] = {
                        "supported": True,
        "success": True,
                        "encapp_result": jsonfile,
                        "mediafile": mediafile
                    }
                else:
                    mode_support[mode_name] = {
                        "supported": False,
                        "success": False,
                        "error": "encapp failed"
                    }
                    
            except Exception as e:
                mode_support[mode_name] = {
                    "supported": False,
                    "success": False,
                    "error": str(e)
                }
        
        test_data["mode_support"] = mode_support
        test_data["supported_modes"] = [mode for mode, info in mode_support.items() if info["supported"]]
        test_data["unsupported_modes"] = [mode for mode, info in mode_support.items() if not info["supported"]]
        
        # Overall success if at least one mode is supported
        overall_success = len(test_data["supported_modes"]) > 0
        
        return {
            "success": overall_success,
            "message": f"Mode support test completed: {len(test_data['supported_modes'])}/{len(bitrate_modes)} modes supported",
            "output_files": output_files,
            "test_data": test_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        "test_data": test_data
    }




def test_bitrate_ladder_vbr(device, input_file, workdir, test_data):
    """Test VBR bitrate ladder using bitrate range."""
    print(f"Running VBR ladder test on {device['encoder']}")
    
    if not encapp:
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    # Initialize test data
    test_data["test_name"] = "bitrate_ladder_vbr"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        # Define bitrate range: 1M-16M-2M means 1M, 3M, 5M, 7M, 9M, 11M, 13M, 15M
        bitrate_range = "1M-16M-2M"
        test_data["bitrate_range"] = bitrate_range
        
        # Initialize test data
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        # Setup test for MP4 input with device decoding
        ava_common.setup_test_for_mp4_input(
            test,
            input_file,
            device,
            "/tmp/",  # mediastore (host path)
        )

        # Enable device decoding for MP4 transcoding
        test.input.device_decode = True
        test.configure.surface = True

        # Create test ID
        test.common.id = f"{__name__}.vbr_ladder.{test.common.id}"
        test.common.description = f"VBR ladder test with range {bitrate_range}"
        test.common.output_filename = test.common.id

        # Configure the test
        test.input.realtime = False
        test.configure.codec = device["encoder"]
        test.configure.bitrate = bitrate_range
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

        # Add to test suite
        test_suite.test.extend([test])
        pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
        encapp.configfile_write(test_suite, pbtxt_file)
        testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name
        
        # Run the test using CLI
        success, output_files = ava_common.run_encapp_cli(pbtxt_file, device, workdir, "/tmp/")
        
        if success:
            test_data["successful_tests"] = 1
            test_data["total_tests"] = 1
            
            # Run encapp_quality on all output files
            quality_csv = ava_common.run_encapp_quality(output_files, workdir)
            
            return {
                "success": True,
                "message": f"VBR ladder test completed with bitrate range {bitrate_range}",
                "output_files": output_files,
                "quality_csv": quality_csv,
                "test_data": test_data
            }
        else:
            return {
                "success": False,
                "error": "encapp CLI failed",
                "test_data": test_data
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
    }

def test_bitrate_ladder_cbr(device, input_file, workdir, test_data):
    """Test CBR bitrate ladder using bitrate range."""
    print(f"Running CBR ladder test on {device['encoder']}")
    print(f"DEBUG: Input file: {input_file}")
    print(f"DEBUG: Workdir: {workdir}")
    print(f"DEBUG: Device: {device}")
    
    if not encapp:
        print("DEBUG: encapp not available")
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    print("DEBUG: encapp is available")
    
    # Initialize test data
    test_data["test_name"] = "bitrate_ladder_cbr"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        # Define bitrate range: 1M-16M-2M means 1M, 3M, 5M, 7M, 9M, 11M, 13M, 15M
        bitrate_range = "1M-16M-2M"
        test_data["bitrate_range"] = bitrate_range
        print(f"DEBUG: Bitrate range set to: {bitrate_range}")
        
        # Initialize test data
        print("DEBUG: Initializing test data")
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        print("DEBUG: Setting up test for input file")
        # Setup test for YUV input (fallback when MP4 transcoding fails)
        ava_common.setup_test_for_input_file(
            test,
            input_file,
            device,
            "_mediastore",  # mediastore (host path)
        )

        # Enable device decoding for MP4 transcoding
        # This is the fallbakc if the mp4 transcoding fails. It seems to be difficult for many devices to do this though
        #test.input.device_decode = True
        #test.configure.surface = True

        # Create test ID
        test.common.id = f"cbr_ladder.[input.filepath]@[configure.bitrate].[confgure.codec].[configure.bitrate].[configure.bitrate_mode]"
        test.common.description = f"CBR ladder test with range {bitrate_range}"
        test.common.output_filename = test.common.id

        # Configure the test
        test.input.realtime = False
        test.configure.codec = device["encoder"]
        test.configure.bitrate = bitrate_range
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.cbr
        print(f"DEBUG: Test configured - codec: {device['encoder']}, bitrate: {bitrate_range}")

        # Add to test suite
        test_suite.test.extend([test])
        pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
        print(f"DEBUG: Writing pbtxt file: {pbtxt_file}")
        encapp.configfile_write(test_suite, pbtxt_file)
        testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name
        print(f"DEBUG: pbtxt file written successfully")
        
        # Run the test using CLI
        print("DEBUG: About to call ava_common.run_encapp_cli")
        success, output_files = ava_common.run_encapp_cli(pbtxt_file, device, workdir, "/tmp/")
        print(f"DEBUG: run_encapp_cli returned: success={success}, output_files={output_files}")
        
        if success:
            test_data["successful_tests"] = 1
            test_data["total_tests"] = 1
            
            # Run encapp_quality on all output files
            quality_csv = ava_common.run_encapp_quality(output_files, workdir)
            
            return {
                "success": True,
                "message": f"CBR ladder test completed with bitrate range {bitrate_range}",
                "output_files": output_files,
                "quality_csv": quality_csv,
                "test_data": test_data
            }
        else:
            return {
                "success": False,
                "error": "encapp CLI failed",
                "test_data": test_data
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
    }

def test_bitrate_ladder_cq(device, input_file, workdir, test_data):
    """Test CQ bitrate ladder across different quality levels."""
    print(f"Running CQ ladder test on {device['encoder']}")
    
    if not encapp:
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    # Initialize test data
    test_data["test_name"] = "bitrate_ladder_cq"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        # Define quality ladder (different CRF/CQ values)
        # Lower values = higher quality, higher bitrate
        quality_levels = [18, 23, 28, 33, 38, 43]
        
        test_data["quality_ladder"] = quality_levels
        ladder_results = []
        output_files = []
        
        for i, quality in enumerate(quality_levels):
            print(f"  Testing CQ at quality level {quality}")
            
            # Initialize test data for this quality level
            testdata = ava_common.initialize_testdata()
            test_suite = encapp.tests_definitions.TestSuite()
            test = encapp.tests_definitions.Test()

            # Setup test for MP4 input with device decoding
            ava_common.setup_test_for_mp4_input(
                test,
                input_file,
                device,
                "/tmp/",  # mediastore (host path)
            )

            # Enable device decoding for MP4 transcoding
            test.input.device_decode = True
            test.configure.surface = True

            # Create unique test ID
            test.common.id = f"{__name__}.cq_{quality}.{test.common.id}"
            test.common.description = f"CQ test at quality level {quality}"
            test.common.output_filename = test.common.id

            # Configure the test
            test.input.realtime = False
            test.configure.codec = device["encoder"]
            test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.cq
            
            # Add CRF/CQ parameter based on encoder
            if "h264" in device["encoder"].lower() or "avc" in device["encoder"].lower():
                params = [
                    encapp.tests_definitions.Parameter(
                        key="video-qp-i",
                        type=encapp.tests_definitions.intType,
                        value=str(quality),
                    ),
                    encapp.tests_definitions.Parameter(
                        key="video-qp-p",
                        type=encapp.tests_definitions.intType,
                        value=str(quality),
                    ),
                ]
            elif "h265" in device["encoder"].lower() or "hevc" in device["encoder"].lower():
                params = [
                    encapp.tests_definitions.Parameter(
                        key="video-qp-i",
                        type=encapp.tests_definitions.intType,
                        value=str(quality),
                    ),
                    encapp.tests_definitions.Parameter(
                        key="video-qp-p",
                        type=encapp.tests_definitions.intType,
                        value=str(quality),
                    ),
                ]
            else:
                # Generic CRF parameter
                params = [
                    encapp.tests_definitions.Parameter(
                        key="crf",
                        type=encapp.tests_definitions.intType,
                        value=str(quality),
                    ),
                ]
            
            test.configure.parameter.extend(params)

            # Add to test suite
            test_suite.test.extend([test])
            pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
            encapp.configfile_write(test_suite, pbtxt_file)
            testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

            # Run the test
            result = encapp.run_codec_tests(
                test_suite,
                [],
                "na",
                device["serial"],
                "/tmp/",  # mediastore (host path)
                workdir,
                device_workdir=device["device_workdir"],
                ignore_results=False,
                fast_copy=True,
                split=False,
                debug=0,
            )
            
            if result[0]:  # Success
                # encapp returns a list of output files: [json_file, mp4_file]
                output_files.extend(result[1])
                
                # Find the JSON and MP4 files in the output
                json_file = None
                mp4_file = None
                for file_path in result[1]:
                    if file_path.endswith('.json'):
                        json_file = file_path
                    elif file_path.endswith('.mp4'):
                        mp4_file = file_path
                
                if json_file and mp4_file:
                    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(json_file).name
                    testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mp4_file).name
                    
                    # Parse the JSON result for this specific test
                    with open(json_file, "r") as f:
                        jsonfile = json.load(f)
                    
                    ladder_results.append({
                        "quality": quality,
                        "success": True,
                        "encapp_result": jsonfile,
                        "json_file": json_file,
                        "mp4_file": mp4_file
                    })
                else:
                    ladder_results.append({
                        "quality": quality,
                        "success": False,
                        "error": "Missing JSON or MP4 output file"
                    })
            else:
                ladder_results.append({
                    "quality": quality,
                    "success": False,
                    "error": "encapp failed"
                })
        
        test_data["ladder_results"] = ladder_results
        test_data["successful_tests"] = len([r for r in ladder_results if r["success"]])
        test_data["total_tests"] = len(ladder_results)
        
        return {
            "success": True,
            "message": f"CQ ladder test completed: {test_data['successful_tests']}/{test_data['total_tests']} successful",
            "output_files": output_files,
            "test_data": test_data,
            "ladder_results": ladder_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
    }

def test_bitrate_ladder_qprange(device, input_file, workdir, test_data):
    """Test QP Range bitrate ladder across different QP ranges."""
    print(f"Running QP Range ladder test on {device['encoder']}")
    
    if not encapp:
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    # Initialize test data
    test_data["test_name"] = "bitrate_ladder_qprange"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        # Define QP range ladder (different QP min/max ranges)
        qp_ranges = [
            {"min": 18, "max": 28},  # High quality, narrow range
            {"min": 20, "max": 35},  # Medium-high quality, medium range
            {"min": 22, "max": 42},  # Medium quality, wide range
            {"min": 25, "max": 45},  # Medium-low quality, wide range
            {"min": 28, "max": 48},  # Low quality, wide range
            {"min": 30, "max": 51},  # Very low quality, full range
        ]
        
        test_data["qp_range_ladder"] = qp_ranges
        ladder_results = []
        output_files = []
        
        for i, qp_range in enumerate(qp_ranges):
            qp_min, qp_max = qp_range["min"], qp_range["max"]
            print(f"  Testing QP Range [{qp_min}, {qp_max}]")
            
            # Initialize test data for this QP range
            testdata = ava_common.initialize_testdata()
            test_suite = encapp.tests_definitions.TestSuite()
            test = encapp.tests_definitions.Test()

            # Setup test for MP4 input with device decoding
            ava_common.setup_test_for_mp4_input(
                test,
                input_file,
                device,
                "/tmp/",  # mediastore (host path)
            )

            # Enable device decoding for MP4 transcoding
            test.input.device_decode = True
            test.configure.surface = True

            # Create unique test ID
            test.common.id = f"{__name__}.qprange_{qp_min}_{qp_max}.{test.common.id}"
            test.common.description = f"QP Range test [{qp_min}, {qp_max}]"
            test.common.output_filename = test.common.id

            # Configure the test
            test.input.realtime = False
            test.configure.codec = device["encoder"]
            test.configure.bitrate = "1Mbps"  # Use fixed bitrate for QP range testing
            test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr
            
            # Add QP range parameters
            params = [
                encapp.tests_definitions.Parameter(
                    key="video-qp-i-min",
                    type=encapp.tests_definitions.intType,
                    value=str(qp_min),
                ),
                encapp.tests_definitions.Parameter(
                    key="video-qp-p-min",
                    type=encapp.tests_definitions.intType,
                    value=str(qp_min),
                ),
                encapp.tests_definitions.Parameter(
                    key="video-qp-i-max",
                    type=encapp.tests_definitions.intType,
                    value=str(qp_max),
                ),
                encapp.tests_definitions.Parameter(
                    key="video-qp-p-max",
                    type=encapp.tests_definitions.intType,
                    value=str(qp_max),
                ),
            ]
            
            test.configure.parameter.extend(params)

            # Add to test suite
            test_suite.test.extend([test])
            pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
            encapp.configfile_write(test_suite, pbtxt_file)
            testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

            # Run the test
            result = encapp.run_codec_tests(
                test_suite,
                [],
                "na",
                device["serial"],
                "/tmp/",  # mediastore (host path)
                workdir,
                device_workdir=device["device_workdir"],
                ignore_results=False,
                fast_copy=True,
                split=False,
                debug=0,
            )
            
            if result[0]:  # Success
                # encapp returns a list of output files: [json_file, mp4_file]
                output_files.extend(result[1])
                
                # Find the JSON and MP4 files in the output
                json_file = None
                mp4_file = None
                for file_path in result[1]:
                    if file_path.endswith('.json'):
                        json_file = file_path
                    elif file_path.endswith('.mp4'):
                        mp4_file = file_path
                
                if json_file and mp4_file:
                    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(json_file).name
                    testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mp4_file).name
                    
                    # Parse the JSON result for this specific test
                    with open(json_file, "r") as f:
                        jsonfile = json.load(f)
                    
                    ladder_results.append({
                        "qp_range": qp_range,
                        "success": True,
                        "encapp_result": jsonfile,
                        "json_file": json_file,
                        "mp4_file": mp4_file
                    })
                else:
                    ladder_results.append({
                        "qp_range": qp_range,
                        "success": False,
                        "error": "Missing JSON or MP4 output file"
                    })
            else:
                ladder_results.append({
                    "qp_range": qp_range,
                    "success": False,
                    "error": "encapp failed"
                })
        
        test_data["ladder_results"] = ladder_results
        test_data["successful_tests"] = len([r for r in ladder_results if r["success"]])
        test_data["total_tests"] = len(ladder_results)
        
        return {
            "success": True,
            "message": f"QP Range ladder test completed: {test_data['successful_tests']}/{test_data['total_tests']} successful",
            "output_files": output_files,
            "test_data": test_data,
            "ladder_results": ladder_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
    }