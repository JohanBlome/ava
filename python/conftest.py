import pytest
import json
import importlib
import os.path
from pathlib import Path
import sys
import datetime
import re

SCRIPT_DIR = Path(globals().get("__file__", "./_")).absolute().parent
TEST_INPUT_MP4 = f"{SCRIPT_DIR}/../vid/johnny.1280x720.60fps.264.mp4"
# allow importing from subprojects
lib_path = os.path.join(
    os.path.dirname(__file__), SCRIPT_DIR, "..", "lib", "encapp", "scripts"
)
sys.path.insert(0, lib_path)
sys.path.insert(0, os.path.dirname(__file__))
ava_common = importlib.import_module("ava-common")
ava_tests = importlib.import_module("ava-tests")
ava_version = importlib.import_module("ava-version")

import encapp  # noqa: E402


all_test_results = {}
output_dir = ""


def is_match(name, encoder, partial=False):
    if partial:
        return name.lower() in encoder["name"].lower()
    else:
        return name.lower() == encoder["name"].lower()


# Get devices and settings:
def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames or "input_file" in metafunc.fixturenames:
        # Access the global_setup fixture manually
        global_setup = metafunc.config._global_setup_resource
        devices = list(global_setup["devices"])  # or keys if you want serials
        input_files = global_setup["input_files"]

        if "device" in metafunc.fixturenames and "input_file" in metafunc.fixturenames:
            argvalues = [
                (device, input_file) for device in devices for input_file in input_files
            ]
            idlist = [
                f"{device['serial']}.{Path(input_file).stem}"
                for device in devices
                for input_file in input_files
            ]
            metafunc.parametrize(("device", "input_file"), argvalues, ids=idlist)
        elif "device" in metafunc.fixturenames:
            metafunc.parametrize("device", devices)
        elif "input_file" in metafunc.fixturenames:
            metafunc.parametrize("input_file", input_files)


def pytest_configure(config):
    """Find all connected devices and prepare a serial list with ne-ded data"""
    debug = 0
    # Find all connected devices
    devices = []

    serials = ava_common.get_all_serials()
    # TODO: how to handle
    assert len(serials) > 0, "No connected devices"

    # Assumption: Encapp is central so we can verify the existance now an check device workdir
    for serial in serials:
        # Name is not great because it will check and install
        ava_common.encapp_is_installed(serial, debug)
        # device workdir
        device_workdir = encapp.get_workdir(serial)
        codecs = ava_common.list_codecs_for_serial(serial)
        # For now we do encoders, let us see if we need somethign else
        encoders = codecs["encoders"]
        enc_conf = config.getoption("--encoder")
        encoder = None
        # Try to lookup the encoder
        simili = [enc for enc in encoders if is_match(enc_conf, enc, True)]
        assert simili, "No matching encoder found"

        encoder = simili[0]["name"]
        dev = {
            "serial": serial,
            "device_workdir": device_workdir,
            "encoder": encoder,
            "encoder_info": simili[0],
        }
        devices.append(dev)

    # Input sources, singel file or folder
    input_files = []
    if config.getoption("--input-folder"):
        folder = Path(config.getoption("--input-folder"))
        if folder.is_dir():
            for file in folder.iterdir():
                m = re.search(r"\.mp4|\.y4m|\.mkv|\.mov", str(file).lower())
                if m:
                    input_files.append(file)
    else:
        input_files.append(config.getoption("--input-file"))

    resource = {"initialized": True, "devices": devices, "input_files": input_files}

    config._global_setup_resource = resource

    # Create a folder to keep results
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    global output_dir
    output_dir = config.getoption("--output-folder")
    if not output_dir:
        output_dir = f"ava-test.{dt_string}"
    os.mkdir(output_dir)


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    resource = {"local_workdir": output_dir}
    yield resource


def pytest_sessionstart(session):
    pass


def pytest_runtest_makereport(item, call):
    if call.when == "call":
        outcome = "passed" if call.excinfo is None else "failed"
        # Retrieve any user properties set by the test
        test_data = dict(item.user_properties) if item.user_properties else {}
        # Store result and data keyed by test name
        all_test_results[item.name] = {"outcome": outcome, "data": test_data}


def pytest_sessionfinish(session, exitstatus):
    # Collect
    if output_dir and len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "ava_aggregated_result.json")
        with open(output_file, "w") as f:
            json.dump(all_test_results, f, indent=2)
    else:
        print("No output directory")


def attach_test_data(item, key, value):
    # Helper for tests to attach arbitrary data to the test item
    item.user_properties.append((key, value))


def pytest_addoption(parser):
    parser.addoption(
        "--keep-per-frame",
        action="store_true",
        default=False,
        help="Keep per-frame data in test output",
    )

    parser.addoption("--input-file", default=TEST_INPUT_MP4, help="")
    parser.addoption("--input-folder", default=None, help="")
    parser.addoption("--output-folder", default=None, help="")
    parser.addoption("--encoder", default="h264", help="")
