#!/usr/bin/env python3

"""
Key frame interval tests for video codecs.
Tests I-frame interval settings and dynamic framerate handling.
"""

from pathlib import Path
import json
import os

# Try to import optional dependencies
try:
    import encapp
    import ava_common
    ENCAPP_AVAILABLE = True
except ImportError:
    ENCAPP_AVAILABLE = False


def test_default_interval(device, input_file, workdir, test_data):
    """
    Test the default I-frame interval and report it.
    
    Args:
        device: Device information dictionary with serial, encoder, etc.
        input_file: Path to input video file
        workdir: Working directory for test files
        test_data: Dictionary to store test results
        
    Returns:
        Dictionary with test results
    """
    if not ENCAPP_AVAILABLE:
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    print(f"Testing default I-frame interval for {device['encoder']}")
    
    # Initialize test data
    test_data["test_name"] = "default_interval"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        # Setup test for input file
        ava_common.setup_test_for_input_file(
            test,
            input_file,
            device,
            "/tmp",  # mediastore
        )

        # Add the actual test name to make it truly unique
        test.common.id = f"{__name__}.{test.common.id}"
        test.common.description = "Test the default i frame interval and report it"
        test.common.output_filename = test.common.id

        # Try to run as fast as possible
        test.input.realtime = False

        # Actual test definition starts here
        test.configure.codec = device["encoder"]
        test.configure.bitrate = "1Mbps"
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

        # Add to test suite
        test_suite.test.extend([test])
        # Write the test to file since it will have an anonymized name otherwise
        pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
        encapp.configfile_write(test_suite, pbtxt_file)
        testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name
        
        # Actual test run
        result = encapp.run_codec_tests(
            test_suite,
            [],
            "na",
            device["serial"],
            "/tmp",  # mediastore
            workdir,
            device_workdir=device["device_workdir"],
            ignore_results=False,
            fast_copy=True,
            split=False,
            debug=1,
        )
        
        # First item contains result
        returncode = result[0]
        if not returncode:
            return {
                "success": False,
                "error": "encapp failed",
                "test_data": test_data
            }

        # We have a result to work with
        output_files = result[1]

        # Verify result
        if len(output_files) == 0:
            return {
                "success": False,
                "error": f"Wrong number of outputs from the test: {len(output_files)}",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name
        
        # Get all frames and verify distance between i frames
        with open(output_files[0], "r") as f:
            jsonfile = json.load(f)

            videofile = jsonfile.get("encodedfile", None)
            if not videofile:
                return {
                    "success": False,
                    "error": "No videofile created",
                    "test_data": test_data
                }

            testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name
            info = encapp.encapp_tool.ffutils.get_video_info(
                f"{workdir}/{videofile}"
            )
            frames = jsonfile["frames"]
            iframes = [frame for frame in frames if frame["iframe"] == 1]
            
            if len(iframes) == 0:
                return {
                    "success": False,
                    "error": "Error in video, no i frames",
                    "test_data": test_data
                }

            # Calculate I-frame interval
            duration = float(info.get("duration", 0))
            interval = duration / len(iframes)
            framerate = float(info.get("framerate", 0))
            
            if framerate <= 0:
                return {
                    "success": False,
                    "error": "Framerate is zero",
                    "test_data": test_data
                }
            
            if interval <= float(1 / framerate):
                return {
                    "success": False,
                    "error": "Key frame only",
                    "test_data": test_data
                }
            
            # Store results
            test_data["duration"] = duration
            test_data["interval"] = interval
            test_data["framerate"] = framerate
            test_data["iframe_count"] = len(iframes)
            test_data["total_frames"] = len(frames)
            test_data["encapp_result"] = testdata

        return {
            "success": True,
            "output_files": [videofile],
            "test_data": test_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
        }


def test_interval_variable_fps(device, input_file, workdir, test_data):
    """
    Set the I-frame interval to 2 and change the framerate, report the average.
    
    Args:
        device: Device information dictionary with serial, encoder, etc.
        input_file: Path to input video file
        workdir: Working directory for test files
        test_data: Dictionary to store test results
        
    Returns:
        Dictionary with test results
    """
    if not ENCAPP_AVAILABLE:
        return {
            "success": False,
            "error": "encapp not available",
            "test_data": test_data
        }
    
    print(f"Testing variable FPS I-frame interval for {device['encoder']}")
    
    # Initialize test data
    test_data["test_name"] = "interval_variable_fps"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        ava_common.setup_test_for_input_file(
            test,
            input_file,
            device,
            "/tmp",  # mediastore
        )

        test.common.id = f"{__name__}.{test.common.id}"
        test.common.description = "Set the i frame interval to 2 and change the framerate, report the average"
        test.common.output_filename = test.common.id
        test.input.realtime = False
        test.configure.codec = device["encoder"]
        test.configure.bitrate = "1Mbps"
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr
        iframe_interval = 2
        test.configure.i_frame_interval = iframe_interval
        
        # Add runtime parameter for dynamic framerate
        runtime = encapp.tests_definitions.Runtime
        parameter = encapp.tests_definitions.Runtime.DynamicFramerateParameter()
        parameter.framenum = 0
        parameter.framerate = float(test.input.framerate) / 2
        test.runtime.dynamic_framerate.extend([parameter])

        test_suite.test.extend([test])

        pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
        encapp.configfile_write(test_suite, pbtxt_file)
        testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

        result = encapp.run_codec_tests(
            test_suite,
            [],
            "na",
            device["serial"],
            "/tmp",  # mediastore
            workdir,
            device_workdir=device["device_workdir"],
            ignore_results=False,
            fast_copy=False,
            split=False,
            debug=0,
        )
        
        # First item contains result
        returncode = result[0]
        if not returncode:
            return {
                "success": False,
                "error": "codec run test failed",
                "test_data": test_data
            }

        output_files = result[1]

        # Verify result
        if len(output_files) == 0:
            return {
                "success": False,
                "error": f"Wrong number of outputs from the test: {len(output_files)}",
                "test_data": test_data
            }
        
        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name

        # Get all frames and verify distance between i frames
        with open(output_files[0], "r") as f:
            jsonfile = json.load(f)

            videofile = jsonfile.get("encodedfile", None)
            if not videofile:
                return {
                    "success": False,
                    "error": "No videofile created",
                    "test_data": test_data
                }

            testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name
            info = encapp.encapp_tool.ffutils.get_video_info(
                f"{workdir}/{videofile}"
            )
            frames = jsonfile["frames"]
            iframes = [frame for frame in frames if frame["iframe"] == 1]

            if len(iframes) == 0:
                return {
                    "success": False,
                    "error": "Error in video, no i frames",
                    "test_data": test_data
                }

            # Calculate I-frame interval
            duration = float(info.get("duration", 0))
            interval = duration / len(iframes)
            framerate = float(info.get("framerate", 0))
            
            if framerate <= 0:
                return {
                    "success": False,
                    "error": "Framerate is zero",
                    "test_data": test_data
                }
            
            if interval <= float(1 / framerate):
                return {
                    "success": False,
                    "error": "Key frame only",
                    "test_data": test_data
                }
            
            # Check if I-frame interval is within expected range (10% tolerance)
            print(f"interval={interval}, framerate={framerate}, duration={duration}")
            tenPct = 0.1 * iframe_interval
            expected_min = iframe_interval - tenPct
            expected_max = iframe_interval + tenPct
            
            if not (interval > expected_min and interval < expected_max):
                return {
                    "success": False,
                    "error": f"I frame interval {interval} not within expected range [{expected_min}, {expected_max}]",
                    "test_data": test_data
                }
            
            # Store results
            test_data["duration"] = duration
            test_data["interval"] = interval
            test_data["framerate"] = framerate
            test_data["iframe_count"] = len(iframes)
            test_data["total_frames"] = len(frames)
            test_data["expected_interval"] = iframe_interval
            test_data["interval_tolerance"] = tenPct
            test_data["encapp_result"] = testdata

        return {
            "success": True,
            "output_files": [videofile],
            "test_data": test_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
        }
