import pathlib
from conftest import attach_test_data
import pytest
import pytest_check
import json
import encapp
import os
import tempfile
from pprint import pprint
import ava_common
import tools.qpextract as qpextract


@pytest.fixture(scope="module")
def local_setup(session_setup):
    global_workdir = session_setup["local_workdir"]
    module_name = pathlib.Path(__file__).stem
    local_workdir = f"{global_workdir}/{module_name}"
    os.mkdir(local_workdir)
    resource = {"local_workdir": local_workdir, "mediastore": "/tmp"}
    yield resource
    # TODO: Add argument for clearance after each module?
    print(f"Exciting {module_name}")


def test_chroma_offset_cbr_hlg(device, input_file, local_setup, request):
    """
    Check qp values for luma and chroma planes using cbr and comparably high bitrates.
    """
    bitrates = ["15M", "60M"]

    for bitrate in bitrates:
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
            device, input_file, local_setup, f"{bitrate}bps", "cbr", __name__, test
        )

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = pathlib.Path(files[0]).name

        jsonfile = None
        with open(files[0], "r") as f:
            jsonfile = json.load(f)

        assert jsonfile, "No valid json data"
        mediafile = f"{local_setup['local_workdir']}/{jsonfile.get('encodedfile')}"
        with pytest_check.check:
            assert mediafile, "No mediafile created"
        testdata[ava_common.DataDefinition.OUTPUT_FILE] = pathlib.Path(mediafile).name

        h265_file = tempfile.NamedTemporaryFile(
            prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
        ).name
        qpextract.extract_simple_stream(mediafile, h265_file)
        # Calculate qp
        stats = qpextract.get_qpstats(h265_file, chroma=True)
        testdata["qpstats"] = stats
        attach_test_data(request.node, jsonfile.get("id"), testdata)


def test_chroma_offset_vbr_hlg(device, input_file, local_setup, request):
    """
    Check qp values for luma and chroma planes using vbr and comparably high bitrates.
    """
    bitrates = ["15M", "60M"]

    for bitrate in bitrates:
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
            device, input_file, local_setup, f"{bitrate}bps", "vbr", __name__, test
        )

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = pathlib.Path(files[0]).name

        jsonfile = None
        with open(files[0], "r") as f:
            jsonfile = json.load(f)

        assert jsonfile, "No valid json data"
        mediafile = f"{local_setup['local_workdir']}/{jsonfile.get('encodedfile')}"
        with pytest_check.check:
            assert mediafile, "No mediafile created"

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
        attach_test_data(request.node, jsonfile.get("id"), testdata)


def test_chroma_offset_qprange_hlg(device, input_file, local_setup, request):
    """
    Check qp values for luma and chroma planes using forced qp ranges using very low values.
    """
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
    test.configure.bitrate = "10Mbps"
    test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr

    # Add to test suite
    test_suite.test.extend([test])

    test.configure.color_range = encapp.tests_definitions.Configure.ColorRange.limited
    test.configure.color_standard = (
        encapp.tests_definitions.Configure.ColorStandard.bt2020
    )
    test.configure.color_transfer = encapp.tests_definitions.Configure.ColorTransfer.hlg

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
    pbtxt_file = f"{local_setup['local_workdir']}/{test.common.id}.pbtxt"
    encapp.configfile_write(test_suite, pbtxt_file)
    files_to_push.append(pbtxt_file)
    testdata[ava_common.DataDefinition.ENCAPP_DEFINITION] = pathlib.Path(
        pbtxt_file
    ).name
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

    returncode = result[0]
    assert returncode, f"error: encapp failed"

    output_files = result[1]  # we should only have one...
    assert output_files and len(output_files) > 0, (
        f"Encapp wrong result: {output_files=}"
    )

    testdata[ava_common.DataDefinition.ENCAPP_RESULT] = pathlib.Path(
        output_files[0]
    ).name
    # Check qp.
    mediafile = None
    jsonfile = None
    with open(output_files[0], "r") as f:
        jsonfile = json.load(f)

    assert jsonfile != None, "Error in encapp data"

    mediafile = f"{local_setup['local_workdir']}/{jsonfile.get('encodedfile')}"
    assert mediafile and len(mediafile) > 0, "no mediafile"

    testdata[ava_common.DataDefinition.OUTPUT_FILE] = pathlib.Path(mediafile).name
    h265_file = tempfile.NamedTemporaryFile(
        prefix=f"ava.{os.path.basename(mediafile)}.", suffix=".265"
    ).name

    qpextract.extract_simple_stream(mediafile, h265_file)
    stats = qpextract.get_qpstats(h265_file, chroma=True)
    testdata["qpstats"] = stats
    attach_test_data(request.node, jsonfile.get("id"), testdata)
