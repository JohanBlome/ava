from pathlib import Path
from conftest import attach_test_data
import pytest
import pytest_check
import json
import encapp
import os
import tempfile
from pprint import pprint
import tools.qpextract as qpextract
import ava_common

qp_settins_working = False


@pytest.fixture(scope="module")
def qptool():
    resources = qpextract.get_qpextract()
    yield resources


@pytest.fixture(scope="module")
def local_setup(session_setup, qptool):
    global_workdir = session_setup["local_workdir"]
    module_name = Path(__file__).stem
    local_workdir = f"{global_workdir}/{module_name}"
    os.mkdir(local_workdir)
    resources = {
        "local_workdir": local_workdir,
        "mediastore": "/tmp",
        "tool_cmd": qptool["cmd"],
        "tool_version": qptool["version"],
    }
    yield resources
    # Teardown. Here we could clear the dvice of all test data to give space for other tests
    # However, we could save tie by keeping test data for other tests as well.
    # TODO: Add argument for clearance after each module?
    print(f"Exciting {module_name}")


def test_qp_simple(device, input_file, local_setup, request):
    """Test the default i frame interval and report it"""

    with pytest_check.check:
        assert local_setup["tool_cmd"] != None, "No qp tool available on path"

    # Check that we have a qp extract cli, currently
    # if local_setup:

    testdata = ava_common.initialize_testdata()

    test_suite = encapp.tests_definitions.TestSuite()
    test = encapp.tests_definitions.Test()

    files_to_push = []
    # Looks at the source fiel and sets props accordingly. We need unique files.
    ava_common.setup_test_for_input_file(
        test,
        input_file,
        device,
        local_setup["mediastore"],
        files_to_push,
    )

    # Add the actual test name to make it truly unique
    test.common.id = f"{__name__}.reference.{test.common.id}"
    test.common.description = f"{__doc__}"
    test.common.output_filename = test.common.id

    # Try to run as fast as posible
    test.input.realtime = False

    # Actual test defintion starts here
    # We will: do two runs, one to check default qp range and then a test to when it is adjusted.
    # Depending on the result we will run further tests or bail.

    test.configure.codec = device["encoder"]
    # TODO: How to handle bitrate, use some kind of good guess based on resolution?<D-s>
    test.configure.bitrate = "1Mbps"
    # TODO: what should be used in the general case?
    test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

    # Add to test suite
    test_suite.test.extend([test])
    # Write the test to file since it will have an anonymized name otherwise
    pbtxt_file = f"{local_setup['local_workdir']}/{test.common.id}.pbtxt"
    encapp.configfile_write(test_suite, pbtxt_file)
    testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

    # Actual test run.
    result = encapp.run_codec_tests(
        test_suite,
        files_to_push,
        "na",  # Should we set something here?
        device["serial"],
        local_setup["mediastore"],
        local_setup["local_workdir"],
        device_workdir=device["device_workdir"],
        ignore_results=False,
        # TODO: should we make this configurable or just clear before running.
        # We could also clear the device after each module run
        fast_copy=True,
        split=False,
        debug=0,
    )
    # First item contains result
    returncode = result[0]
    assert returncode, f"error: encapp failed"

    output_files = result[1]  # we should only have one...
    assert output_files and len(output_files) > 0, "Wrong results from reference run"

    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name
    # Check qp.
    mediafile = None
    jsonfile = None
    with open(output_files[0], "r") as f:
        jsonfile = json.load(f)

    if jsonfile == None:
        # TODO: early return
        pass

    mediafile = f"{local_setup['local_workdir']}/{jsonfile.get('encodedfile')}"
    assert mediafile and len(mediafile) > 0, "no mediafile"

    testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mediafile).name
    h265_file = tempfile.NamedTemporaryFile(
        prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
    ).name

    qpextract.extract_simple_stream(mediafile, h265_file)
    stats = qpextract.get_qpstats(h265_file)
    testdata["qpstats"] = stats
    # Attach reference
    attach_test_data(request.node, jsonfile.get("id"), testdata)
    # Second run
    testdata = ava_common.initialize_testdata()
    test_suite = encapp.tests_definitions.TestSuite()
    test = encapp.tests_definitions.Test()

    files_to_push = []
    # Looks at the source fiel and sets props accordingly. We need unique files.
    ava_common.setup_test_for_input_file(
        test,
        input_file,
        device,
        local_setup["mediastore"],
        files_to_push,
    )

    # Add the actual test name to make it truly unique
    test.common.id = f"{__name__}.{test.common.id}"
    test.common.description = f"{__doc__}"
    test.common.output_filename = test.common.id

    # Try to run as fast as posible
    test.input.realtime = False

    # Now for the second run that is depending on the first
    test.configure.codec = device["encoder"]
    # TODO: How to handle bitrate, use some kind of good guess based on resolution?<D-s>
    test.configure.bitrate = "1Mbps"
    # TODO: what should be used in the general case?
    test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

    # Add to test suite
    test_suite.test.extend([test])
    # Write the test to file since it will have an anonymized name otherwise
    pbtxt_file = f"{local_setup['local_workdir']}/{test.common.id}.pbtxt"
    encapp.configfile_write(test_suite, pbtxt_file)
    files_to_push.append(pbtxt_file)
    testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

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
    # Actual test run.
    result = encapp.run_codec_tests(
        test_suite,
        files_to_push,
        "na",  # Should we set something here?
        device["serial"],
        local_setup["mediastore"],
        local_setup["local_workdir"],
        device_workdir=device["device_workdir"],
        ignore_results=False,
        # TODO: should we make this configurable or just clear before running.
        # We could also clear the device after each module run
        fast_copy=True,
        split=False,
        debug=1,
    )
    # First item contains result
    returncode = result[0]
    assert returncode, f"error: encapp failed"

    output_files = result[1]  # we should only have one...
    assert output_files and len(output_files) > 0, (
        f"Encapp wrong result: {output_files=}"
    )

    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name
    # Check qp.
    mediafile = None
    jsonfile = None
    with open(output_files[0], "r") as f:
        jsonfile = json.load(f)

    assert jsonfile != None, "Error in encapp data"

    mediafile = f"{local_setup['local_workdir']}/{jsonfile.get('encodedfile')}"
    assert mediafile and len(mediafile) > 0, "no mediafile"

    testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(mediafile).name
    h265_file = tempfile.NamedTemporaryFile(
        prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
    ).name

    qpextract.extract_simple_stream(mediafile, h265_file)
    stats = qpextract.get_qpstats(h265_file)
    testdata["qpstats"] = stats
    # Attach reference
    attach_test_data(request.node, jsonfile.get("id"), testdata)
    global qp_settins_working
    qp_settins_working = True

    assert qpmin_2nd >= qpmin + offset and qpmax_2nd <= qpmax + offset, "Wrong qp range"


def test_qp_bound_ladder(device, input_file, local_setup, request):
    pprint(f"Ladder: {local_setup=}")

    assert local_setup["tool_cmd"] != None, "No qp tool available on path"
    assert qp_settins_working
    print("Run it")


def test_some_test(device, input_file, local_setup, request):
    print("Some test")
    pprint(request)
    pass
