from pathlib import Path
from conftest import attach_test_data
import pytest
import pytest_check
import json
import encapp
import os
import importlib

ava_common = importlib.import_module("ava-common")

vbr_support = False
cbr_support = False
cq_support = False


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


# Define pixelrates and bitrates in kbps
pixelrates = {1920 * 1080 * 30: 8000, 1920 * 1080 * 60: 10000, 1280 * 720 * 30: 6000}


def find_bitrate_for_video(videopath: str, encoder: str) -> str:
    info = encapp.encapp_tool.ffutils.get_video_info(videopath)
    pixelrate = info["width"] * info["height"] * info["framerate"]
    # hevc
    if "hevc" in encoder.lower() or "h265" in encoder.lower():
        # Use avc as ref, hevc is 20% better?
        pixelrate = int(0.8 * pixelrate)

    bitrate = -1
    higher = -1
    for pixelrate_ in pixelrates:
        if pixelrate_ > pixelrate:
            higher = pixelrate_
            break

    ratio = pixelrate / higher
    bitrate = int(ratio * pixelrates[higher])
    return f"{bitrate}kbps"


def test_bitrate_mode_support(device, input_file, local_setup, request):
    """
    Decoder will contain information about the supported bitrate modes,
    check that they do not throw an error and that they ar set

                 'encoder_capabilities': {'MODE_CBR': True,
                                          'MODE_CQ': False,
                                          'MODE_VBR': True,
                                          'complexity_range': '[0, 0]',
                                          'quality_range': '[0, 0]'},
    """
    encoder_info = device.get("encoder_info", None)
    assert encoder_info
    mt = encoder_info.get("media_type", None)
    assert mt, "No mediatype"
    caps = mt.get("encoder_capabilities", None)
    assert caps, "No caps found for the codec"

    global vbr_support, cbr_support, cq_support
    vbr_support = bool(caps.get("MODE_VBR"))
    cbr_support = bool(caps.get("MODE_CBR"))
    cq_support = bool(caps.get("MODE_CQ"))

    # Find suitable bitrate for the video content
    if vbr_support:
        bitrate = find_bitrate_for_video(input_file, device["encoder"])
        files, testdata = ava_common.run_capture(
            device, input_file, local_setup, bitrate, "vbr", __name__
        )

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(files[0]).name

        jsonfile = None
        with open(files[0], "r") as f:
            jsonfile = json.load(f)

        assert jsonfile, "No valid json data"
        videofile = jsonfile.get("encodedfile", None)
        with pytest_check.check:
            assert videofile, "No videofile created"

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name
        # Check that the mediacodec setting is matching vbr
        # TODO

        attach_test_data(request.node, jsonfile.get("id"), testdata)

    if cbr_support:
        bitrate = find_bitrate_for_video(input_file, device["encoder"])
        files, testdata = ava_common.run_capture(
            device, input_file, local_setup, bitrate, "cbr", __name__
        )

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(files[0]).name

        jsonfile = None
        with open(files[0], "r") as f:
            jsonfile = json.load(f)

        assert jsonfile, "No valid json data"
        videofile = jsonfile.get("encodedfile", None)
        with pytest_check.check:
            assert videofile, "No videofile created"

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name
        # Check that the mediacodec setting is matching vbr
        # TODO

        attach_test_data(request.node, jsonfile.get("id"), testdata)

    # TODO: cq


def test_bitrate_ladder_vbr(device, input_file, local_setup, request):
    # We can have a bunch of strategies for this, but for now just run with a simple solution

    bitrate = int(find_bitrate_for_video(input_file, device["encoder"])[:-4])
    bitrates = []
    if bitrate >= 1000:
        bitrate = int((bitrate // 1000) * 1e6)
        # 4 below and 4 above
        inc = int(bitrate // 10)
        for bt in range(bitrate - int(4 * inc), bitrate, inc):
            bitrates.append(bt)
        inc = int(bitrate // 2)
        for bt in range(bitrate, bitrate + int(4 * inc), inc):
            bitrates.append(bt)
    else:
        bitrate = int((bitrate // 100) * 1e5)
        if bitrate == 0:
            bitrate = 100000  # 100kbps
        # 4 below and 4 above
        inc = int(bitrate // 10)
        for bt in range(bitrate - int(4 * inc), bitrate, inc):
            bitrates.append(bt)
        inc = int(bitrate // 2)
        for bt in range(bitrate, bitrate + int(4 * inc), inc):
            bitrates.append(bt)

    for bitrate in bitrates:
        files, testdata = ava_common.run_capture(
            device, input_file, local_setup, f"{bitrate}bps", "vbr", __name__
        )

        testdata[ava_common.DataDefinition.ENCAPP_RESULT] = Path(files[0]).name

        jsonfile = None
        with open(files[0], "r") as f:
            jsonfile = json.load(f)

        assert jsonfile, "No valid json data"
        videofile = jsonfile.get("encodedfile", None)
        with pytest_check.check:
            assert videofile, "No videofile created"

        testdata[ava_common.DataDefinition.OUTPUT_FILE] = Path(videofile).name

        attach_test_data(request.node, jsonfile.get("id"), testdata)
