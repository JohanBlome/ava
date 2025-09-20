from pathlib import Path
from conftest import attach_test_data
import pytest
import pytest_check
import json
import encapp
import os
import ava_common


@pytest.fixture(scope="module")
def local_setup(session_setup):
    global_workdir = session_setup["local_workdir"]
    module_name = Path(__file__).stem
    local_workdir = f"{global_workdir}/{module_name}"
    os.mkdir(local_workdir)
    resource = {"local_workdir": local_workdir, "mediastore": "/tmp"}
    yield resource
    # Teardown. Here we could clear the dvice of all test data to give space for other tests
    # However, we could save tie by keeping test data for other tests as well.
    # TODO: Add argument for clearance after each module?
    print(f"Exciting {module_name}")


def test_default_interval(device, input_file, local_setup, request):
    """Test the default i frame interval and report it"""
    # Data to be reported
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

    # Actual test defintion starts here
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
        debug=1,
    )
    # First item contains result
    returncode = result[0]
    assert returncode, f"error: encapp failed"

    # We have a result to work with
    output_files = result[1]

    # VERIFY result
    assert len(output_files) > 0, (
        f"Wrong number of outputs from the test: {len(output_files)}"
    )

    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name
    # Get all frames and verify distance between i frames
    jsonfile = None
    with open(output_files[0], "r") as f:
        jsonfile = json.load(f)

        videofile = jsonfile.get("encodedfile", None)
        with pytest_check.check:
            assert videofile, "No videofile created"

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name
        info = encapp.encapp_tool.ffutils.get_video_info(
            f"{local_setup['local_workdir']}/{videofile}"
        )
        frames = jsonfile["frames"]
        iframes = [frame for frame in frames if frame["iframe"] == 1]
        with pytest_check.check:
            assert len(iframes) > 0, "Error in video, no i frames"

        # Let us do it in the easy way
        duration = float(info.get("duration", 0))
        interval = duration / len(iframes)
        framerate = float(info.get("framerate", 0))
        # TODO: we could check the intra distance as well.
        with pytest_check.check:
            assert framerate > 0, "Framerate is zero"
        with pytest_check.check:
            assert interval > float(1 / framerate), "Key fram only"

    # report data
    attach_test_data(request.node, test.common.id, testdata)
    # cleanup


def test_interval_variable_fps(device, input_file, local_setup, request):
    """Set the i frame interval to 2 and change the framerate, report the average"""
    # Data to be reported
    testdata = ava_common.initialize_testdata()

    test_suite = encapp.tests_definitions.TestSuite()
    test = encapp.tests_definitions.Test()

    files_to_push = []
    ava_common.setup_test_for_input_file(
        test,
        input_file,
        device,
        local_setup["mediastore"],
        files_to_push,
    )

    test.common.id = f"{__name__}.{test.common.id}"
    test.common.description = f"{__doc__}"
    test.common.output_filename = test.common.id
    test.input.realtime = False
    test.configure.codec = device["encoder"]
    test.configure.bitrate = "1Mbps"
    test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr
    iframe_interval = 2
    test.configure.i_frame_interval = iframe_interval
    # Add runtime parameter
    runtime = encapp.tests_definitions.Runtime
    parameter = encapp.tests_definitions.Runtime.DynamicFramerateParameter()
    parameter.framenum = 0
    parameter.framerate = float(test.input.framerate) / 2
    test.runtime.dynamic_framerate.extend([parameter])

    test_suite.test.extend([test])

    pbtxt_file = f"{local_setup['local_workdir']}/{test.common.id}.pbtxt"
    encapp.configfile_write(test_suite, pbtxt_file)
    testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = Path(pbtxt_file).name

    result = encapp.run_codec_tests(
        test_suite,
        files_to_push,
        "na",  # Should we set something here?
        device["serial"],
        local_setup["mediastore"],
        local_setup["local_workdir"],
        device_workdir=device["device_workdir"],
        ignore_results=False,
        fast_copy=False,
        split=False,
        debug=0,
    )
    # First item contains result
    returncode = result[0]
    assert returncode, f"error: codec run test"

    output_files = result[1]

    # VERIFY result
    assert len(output_files) > 0, (
        f"Wrong number of outputs from the test: {len(output_files)}"
    )
    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(output_files[0]).name

    # Get all frames and verify distance between i frames
    jsonfile = None
    with open(output_files[0], "r") as f:
        jsonfile = json.load(f)

        videofile = jsonfile.get("encodedfile", None)
        with pytest_check.check:
            assert videofile, "No videofile created"

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name
        info = encapp.encapp_tool.ffutils.get_video_info(
            f"{local_setup['local_workdir']}/{videofile}"
        )
        frames = jsonfile["frames"]
        iframes = [frame for frame in frames if frame["iframe"] == 1]

        with pytest_check.check:
            assert len(iframes) > 0, "Error in video, no i frames"

        # Let us do it in the easy way
        duration = float(info.get("duration", 0))
        interval = duration / len(iframes)
        framerate = float(info.get("framerate", 0))
        # TODO: we could check the intra distance as well.
        with pytest_check.check:
            assert framerate > 0, "Framerate is zero"
        with pytest_check.check:
            assert interval > float(1 / framerate), "Key fram only"
        # Accept 10%, TODO: should be defined somewhere
        print(f"{interval=}, {framerate=}, {duration=}")
        tenPct = 0.1 * iframe_interval
        with pytest_check.check:
            assert (
                interval > iframe_interval - tenPct
                and interval < iframe_interval + tenPct
            ), "I frame interval defined as frame count in configuration"

    # report data
    attach_test_data(request.node, test.common.id, testdata)
    # cleanup
