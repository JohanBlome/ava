#!/usr/bin/env python3

"""
Chroma offset tests for video codecs.
Tests QP values for luma and chroma planes using different bitrate modes.
"""

import pathlib
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


def test_chroma_offset_cbr_hlg(device, input_file, workdir, test_data):
    """
    Check QP values for luma and chroma planes using CBR and comparably high bitrates.
    
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
    
    print(f"Testing chroma offset CBR HLG for {device['encoder']}")
    
    # Initialize test data
    test_data["test_name"] = "chroma_offset_cbr_hlg"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    bitrates = ["15M", "60M"]
    test_data["bitrates"] = bitrates
    output_files = []
    bitrate_results = []
    
    for bitrate in bitrates:
        try:
            test = encapp.tests_definitions.Test()
            test.configure.color_range = (
                encapp.tests_definitions.Configure.ColorRange.limited
            )
            test.configure.color_standard = (
                encapp.tests_definitions.Configure.ColorStandard.bt2020
            )
            test.configure.color_transfer = (
                encapp.tests_definitions.Configure.ColorTransfer.hlg
            )
            
            files, testdata = ava_common.run_capture(
                device, input_file, {"local_workdir": workdir, "mediastore": "/tmp"}, 
                f"{bitrate}bps", "cbr", __name__, test
            )

            testdata[ava_common.DataDefinition.ENCAPP_RESULT] = pathlib.Path(files[0]).name

            with open(files[0], "r") as f:
                jsonfile = json.load(f)

            if not jsonfile:
                bitrate_results.append({
                    "bitrate": bitrate,
                    "error": "No valid json data"
                })
                continue
                
            mediafile = f"{workdir}/{jsonfile.get('encodedfile')}"
            if not mediafile:
                bitrate_results.append({
                    "bitrate": bitrate,
                    "error": "No mediafile created"
                })
                continue
                
            testdata[ava_common.DataDefinition.OUTPUT_FILE] = pathlib.Path(mediafile).name

            h265_file = tempfile.NamedTemporaryFile(
                prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
            ).name
            qpextract.extract_simple_stream(mediafile, h265_file)
            # Calculate qp
            stats = qpextract.get_qpstats(h265_file, chroma=True)
            testdata["qpstats"] = stats
            
            output_files.append(mediafile)
            bitrate_results.append({
                "bitrate": bitrate,
                "testdata": testdata,
                "qpstats": stats,
                "mediafile": mediafile
            })
            
        except Exception as e:
            bitrate_results.append({
                "bitrate": bitrate,
                "error": str(e)
            })
    
    test_data["bitrate_results"] = bitrate_results
    
    return {
        "success": True,
        "output_files": output_files,
        "test_data": test_data
    }


def test_chroma_offset_vbr_hlg(device, input_file, workdir, test_data):
    """
    Check QP values for luma and chroma planes using VBR and comparably high bitrates.
    
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
    
    print(f"Testing chroma offset VBR HLG for {device['encoder']}")
    
    # Initialize test data
    test_data["test_name"] = "chroma_offset_vbr_hlg"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    bitrates = ["15M", "60M"]
    test_data["bitrates"] = bitrates
    output_files = []
    bitrate_results = []
    
    for bitrate in bitrates:
        try:
            test = encapp.tests_definitions.Test()
            test.configure.color_range = (
                encapp.tests_definitions.Configure.ColorRange.limited
            )
            test.configure.color_standard = (
                encapp.tests_definitions.Configure.ColorStandard.bt2020
            )
            test.configure.color_transfer = (
                encapp.tests_definitions.Configure.ColorTransfer.hlg
            )
            
            files, testdata = ava_common.run_capture(
                device, input_file, {"local_workdir": workdir, "mediastore": "/tmp"}, 
                f"{bitrate}bps", "vbr", __name__, test
            )

            testdata[ava_common.DataDefinition.ENCAPP_RESULT] = pathlib.Path(files[0]).name

            with open(files[0], "r") as f:
                jsonfile = json.load(f)

            if not jsonfile:
                bitrate_results.append({
                    "bitrate": bitrate,
                    "error": "No valid json data"
                })
                continue
                
            mediafile = f"{workdir}/{jsonfile.get('encodedfile')}"
            if not mediafile:
                bitrate_results.append({
                    "bitrate": bitrate,
                    "error": "No mediafile created"
                })
                continue

            pprint(files)
            testdata[ava_common.DataDefinition.OUTPUT_FILE] = pathlib.Path(mediafile).name

            pprint(testdata[ava_common.DataDefinition.OUTPUT_FILE])
            pprint(mediafile)
            h265_file = tempfile.NamedTemporaryFile(
                prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
            ).name
            qpextract.extract_simple_stream(mediafile, h265_file)
            # Calculate qp
            stats = qpextract.get_qpstats(h265_file, chroma=True)
            testdata["qpstats"] = stats
            
            output_files.append(mediafile)
            bitrate_results.append({
                "bitrate": bitrate,
                "testdata": testdata,
                "qpstats": stats,
                "mediafile": mediafile
            })
            
        except Exception as e:
            bitrate_results.append({
                "bitrate": bitrate,
                "error": str(e)
            })
    
    test_data["bitrate_results"] = bitrate_results
    
    return {
        "success": True,
        "output_files": output_files,
        "test_data": test_data
    }


def test_chroma_offset_qprange_hlg(device, input_file, workdir, test_data):
    """
    Check QP values for luma and chroma planes using forced QP ranges with very low values.
    
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
    
    print(f"Testing chroma offset QP range HLG for {device['encoder']}")
    
    # Initialize test data
    test_data["test_name"] = "chroma_offset_qprange_hlg"
    test_data["device_serial"] = device["serial"]
    test_data["encoder"] = device["encoder"]
    test_data["input_file"] = input_file
    
    try:
        testdata = ava_common.initialize_testdata()
        test_suite = encapp.tests_definitions.TestSuite()
        test = encapp.tests_definitions.Test()

        files_to_push = []
        # Setup test for input file
        ava_common.setup_test_for_input_file(
            test,
            input_file,
            device,
            "/tmp",  # mediastore
            files_to_push,
        )

        # Add the actual test name to make it truly unique
        test.common.id = f"{__name__}.{test.common.id}"
        test.common.description = "Check qp values for luma and chroma planes using forced qp ranges using very low values"
        test.common.output_filename = test.common.id

        # Try to run as fast as possible
        test.input.realtime = False

        # Configure the test
        test.configure.codec = device["encoder"]
        test.configure.bitrate = "10Mbps"
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

        # Add to test suite
        test_suite.test.extend([test])

        test.configure.color_range = encapp.tests_definitions.Configure.ColorRange.limited
        test.configure.color_standard = (
            encapp.tests_definitions.Configure.ColorStandard.bt2020
        )
        test.configure.color_transfer = encapp.tests_definitions.Configure.ColorTransfer.hlg

        # Add QP range parameters
        params = [
            encapp.tests_definitions.Parameter(
                key="video-qp-i-min",
                type=encapp.tests_definitions.intType,
                value=str(1),
            ),
            encapp.tests_definitions.Parameter(
                key="video-qp-p-min",
                type=encapp.tests_definitions.intType,
                value=str(1),
            ),
            encapp.tests_definitions.Parameter(
                key="video-qp-i-max",
                type=encapp.tests_definitions.intType,
                value=str(12),
            ),
            encapp.tests_definitions.Parameter(
                key="video-qp-p-max",
                type=encapp.tests_definitions.intType,
                value=str(12),
            ),
        ]

        test.configure.parameter.extend(params)
        test_suite.test.extend([test])
        
        # Write the test to file since it will have an anonymized name otherwise
        pbtxt_file = f"{workdir}/{test.common.id}.pbtxt"
        encapp.configfile_write(test_suite, pbtxt_file)
        files_to_push.append(pbtxt_file)
        testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = pathlib.Path(
            pbtxt_file
        ).name
        
        # Actual test run
        result = encapp.run_codec_tests(
            test_suite,
            files_to_push,
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

        returncode = result[0]
        if not returncode:
            return {
                "success": False,
                "error": "encapp failed",
                "test_data": test_data
            }

        output_files = result[1]  # we should only have one...
        if not output_files or len(output_files) == 0:
            return {
                "success": False,
                "error": f"Encapp wrong result: {output_files=}",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = pathlib.Path(
            output_files[0]
        ).name
        
        # Check qp
        with open(output_files[0], "r") as f:
            jsonfile = json.load(f)

        if not jsonfile:
            return {
                "success": False,
                "error": "Error in encapp data",
                "test_data": test_data
            }

        mediafile = f"{workdir}/{jsonfile.get('encodedfile')}"
        if not mediafile or len(mediafile) == 0:
            return {
                "success": False,
                "error": "no mediafile",
                "test_data": test_data
            }

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = pathlib.Path(mediafile).name
        h265_file = tempfile.NamedTemporaryFile(
            prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
        ).name

        qpextract.extract_simple_stream(mediafile, h265_file)
        stats = qpextract.get_qpstats(h265_file, chroma=True)
        testdata["qpstats"] = stats
        
        test_data["qpstats"] = stats
        test_data["encapp_result"] = testdata

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
