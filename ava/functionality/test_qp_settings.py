#!/usr/bin/env python3

"""
QP settings tests for video codecs.
Tests QP range settings and their effects on encoding quality.
"""

from pathlib import Path
import json
import os
import tempfile
from pprint import pprint

# Try to import optional dependencies
try:
    import encapp
    import ava_common
    import tools.qpextract as qpextract
    ENCAPP_AVAILABLE = True
except ImportError:
    ENCAPP_AVAILABLE = False

qp_settings_working = False


def test_qp_simple(device, input_file, workdir, test_data):
    """
    Test QP settings with default and adjusted ranges.
    
    Args:
        device: Device information dictionary with serial, encoder, etc.
        input_file: Path to input video file
        workdir: Working directory for test files
        test_data: Dictionary to store test results
        
    Returns:
        Dictionary with test results
    """
    print(f"Testing QP settings for {device['encoder']}")
    
    # Get mediastore from test_data
    mediastore = test_data.get("mediastore", "_mediastore")
    
    # Initialize test data
    test_data["test_name"] = "qp_simple"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        # First run - reference with default QP settings
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        # Setup test for input file
        ava_common.setup_test_for_input_file(
            test,
            input_file,
            device,
            mediastore,  # mediastore
        )

        # Add the actual test name to make it truly unique
        test.common.id = f"{__name__}.reference.{test.common.id}"
        test.common.description = "Test the default QP settings and report it"
        test.common.output_filename = test.common.id

        # Try to run as fast as possible
        test.input.realtime = False

        # Configure the test
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
            mediastore,  # mediastore
            workdir,
            device_workdir=device["device_workdir"],
            ignore_results=False,
            fast_copy=True,
            split=False,
            debug=0,
        )
        
        # First item contains result
        returncode = result[0]
        if not returncode:
            return {
                "success": False,
                "error": "encapp failed on reference run",
                "test_data": test_data
            }

        output_files = result[1]  # we should only have one...
        if not output_files or len(output_files) == 0:
            return {
                "success": False,
                "error": "Wrong results from reference run",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name
        
        # Check QP from reference run
        with open(output_files[0], "r") as f:
            jsonfile = json.load(f)

        if not jsonfile:
            return {
                "success": False,
                "error": "No valid json data from reference run",
                "test_data": test_data
            }

        mediafile = f"{workdir}/{jsonfile.get('encodedfile')}"
        if not mediafile or len(mediafile) == 0:
            return {
                "success": False,
                "error": "no mediafile from reference run",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mediafile).name
        h265_file = tempfile.NamedTemporaryFile(
            prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
        ).name

        qpextract.extract_simple_stream(mediafile, h265_file)
        stats = qpextract.get_qpstats(h265_file)
        testdata["qpstats"] = stats
        
        # Store reference results
        test_data["reference_qpstats"] = stats
        test_data["reference_encapp_result"] = testdata
        
        # Extract QP range from stats for second run
        qpmin = stats.get("qp_min", 0)
        qpmax = stats.get("qp_max", 51)
        
        # Second run - with adjusted QP settings
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        # Setup test for input file
        ava_common.setup_test_for_input_file(
            test,
            input_file,
            device,
            mediastore,  # mediastore
        )

        # Add the actual test name to make it truly unique
        test.common.id = f"{__name__}.{test.common.id}"
        test.common.description = "Test with adjusted QP settings"
        test.common.output_filename = test.common.id

        # Try to run as fast as possible
        test.input.realtime = False

        # Configure the test
        test.configure.codec = device["encoder"]
        test.configure.bitrate = "1Mbps"
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

        # Add to test suite
        test_suite.test.extend([test])
        # Write the test to file since it will have an anonymized name otherwise
        pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
        encapp.configfile_write(test_suite, pbtxt_file)
        testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

        # Calculate adjusted QP range
        offset = int((qpmax - qpmin) / 4)
        params = [
            encapp.tests_definitions.Parameter(
                key="video-qp-i-min",
                type=encapp.tests_definitions.intType,
                value=str(qpmin + offset),
            ),
            encapp.tests_definitions.Parameter(
                key="video-qp-p-min",
                type=encapp.tests_definitions.intType,
                value=str(qpmin + offset),
            ),
            encapp.tests_definitions.Parameter(
                key="video-qp-i-max",
                type=encapp.tests_definitions.intType,
                value=str(qpmax - offset),
            ),
            encapp.tests_definitions.Parameter(
                key="video-qp-p-max",
                type=encapp.tests_definitions.intType,
                value=str(qpmax - offset),
            ),
        ]

        test.configure.parameter.extend(params)
        test_suite.test.extend([test])
        
        # Actual test run
        result = encapp.run_codec_tests(
            test_suite,
            [],
            "na",
            device["serial"],
            mediastore,  # mediastore
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
                "error": "encapp failed on adjusted QP run",
                "test_data": test_data
            }

        output_files = result[1]  # we should only have one...
        if not output_files or len(output_files) == 0:
            return {
                "success": False,
                "error": f"Encapp wrong result: {output_files=}",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name
        
        # Check QP from adjusted run
        with open(output_files[0], "r") as f:
            jsonfile = json.load(f)

        if not jsonfile:
            return {
                "success": False,
                "error": "Error in encapp data from adjusted run",
                "test_data": test_data
            }

        mediafile = f"{workdir}/{jsonfile.get('encodedfile')}"
        if not mediafile or len(mediafile) == 0:
            return {
                "success": False,
                "error": "no mediafile from adjusted run",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mediafile).name
        h265_file = tempfile.NamedTemporaryFile(
            prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
        ).name

        qpextract.extract_simple_stream(mediafile, h265_file)
        stats = qpextract.get_qpstats(h265_file)
        testdata["qpstats"] = stats
        
        # Store adjusted results
        test_data["adjusted_qpstats"] = stats
        test_data["adjusted_encapp_result"] = testdata
        test_data["qp_range"] = {
            "original_min": qpmin,
            "original_max": qpmax,
            "adjusted_min": qpmin + offset,
            "adjusted_max": qpmax - offset,
            "offset": offset
        }
        
        # Verify QP range was applied correctly
        qpmin_2nd = stats.get("qp_min", 0)
        qpmax_2nd = stats.get("qp_max", 51)
        
        if not (qpmin_2nd >= qpmin + offset and qpmax_2nd <= qpmax - offset):
            return {
                "success": False,
                "error": f"Wrong QP range: expected [{qpmin + offset}, {qpmax - offset}], got [{qpmin_2nd}, {qpmax_2nd}]",
                "test_data": test_data
            }
        
        global qp_settings_working
        qp_settings_working = True
        
        return {
            "success": True,
            "output_files": [mediafile],
            "test_data": test_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
        }


def test_qp_bound_ladder(device, input_file, workdir, test_data):
    """
    Test QP bound ladder with various QP ranges.
    
    Args:
        device: Device information dictionary with serial, encoder, etc.
        input_file: Path to input video file
        workdir: Working directory for test files
        test_data: Dictionary to store test results
        
    Returns:
        Dictionary with test results
    """
    print(f"Testing QP bound ladder for {device['encoder']}")
    
    # Get mediastore from test_data
    mediastore = test_data.get("mediastore", "_mediastore")
    
    # Initialize test data
    test_data["test_name"] = "qp_bound_ladder"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    # Check if QP settings are working from previous test
    if not qp_settings_working:
        return {
            "success": False,
            "error": "QP settings not working from previous test",
            "test_data": test_data
        }
    
    print("Running QP bound ladder test")
    
    # This is a placeholder for the ladder test
    # In a real implementation, this would test various QP ranges
    test_data["ladder_status"] = "placeholder_implemented"
    
    return {
        "success": True,
        "output_files": [],
        "test_data": test_data
    }


def test_some_test(device, input_file, workdir, test_data):
    """
    Example test function.
    
    Args:
        device: Device information dictionary with serial, encoder, etc.
        input_file: Path to input video file
        workdir: Working directory for test files
        test_data: Dictionary to store test results
        
    Returns:
        Dictionary with test results
    """
    print("Running some test")
    
    # Initialize test data
    test_data["test_name"] = "some_test"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    # This is just a placeholder test
    test_data["status"] = "placeholder_implemented"
    
    return {
        "success": True,
        "output_files": [],
        "test_data": test_data
    }
