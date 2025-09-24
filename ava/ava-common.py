#!/usr/bin/env python3
import encapp
import encapp_tool
import os
import re
import subprocess
import sys
import tempfile
import pathlib
from typing import Any
from enum import StrEnum

# allow importing from subprojects
root_path = os.path.join(os.path.dirname(__file__), "..")
lib_path = os.path.join(root_path, "lib", "encapp", "scripts")
sys.path.insert(0, lib_path)
# import encapp  # noqa: E402

ENCODED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
# TODO: figure out how to handle this. On osx the /usr/bin/time is not the GNU version.


def run(command, **kwargs):
    debug = kwargs.get("debug", 0)
    dry_run = kwargs.get("dry_run", False)
    env = kwargs.get("env", None)
    stdin = subprocess.PIPE if kwargs.get("stdin", False) else None
    bufsize = kwargs.get("bufsize", 0)
    universal_newlines = kwargs.get("universal_newlines", False)
    default_close_fds = True if sys.platform == "linux2" else False
    close_fds = kwargs.get("close_fds", default_close_fds)
    shell = kwargs.get("shell", type(command) in (type(""), type("")))
    logfd = kwargs.get("logfd", sys.stdout)
    if debug > 0:
        print(f"$ {command}", file=logfd)
    if dry_run:
        return 0, b"stdout", b"stderr"
    gnu_time = kwargs.get("gnu_time", False)
    if gnu_time:
        # GNU /usr/bin/time support
        command = f"/usr/bin/time -v {command}"

    p = subprocess.Popen(  # noqa: E501
        command,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=bufsize,
        universal_newlines=universal_newlines,
        env=env,
        close_fds=close_fds,
        shell=shell,
    )
    # wait for the command to terminate
    if stdin is not None:
        out, err = p.communicate(stdin)
    else:
        out, err = p.communicate()
    returncode = p.returncode
    # clean up
    del p
    if gnu_time:
        # make sure the stats are there
        GNU_TIME_BYTES = b"\n\tUser time"
        if GNU_TIME_BYTES in err:
            gnu_time_str = err[err.index(GNU_TIME_BYTES) :].decode("ascii")
            gnu_time_stats = gnu_time_parse(gnu_time_str, logfd, debug)
            err = err[0 : err.index(GNU_TIME_BYTES) :]
            return returncode, out, err, gnu_time_stats
    # return results
    return returncode, out, err, None


GNU_TIME_DEFAULT_KEY_DICT = {
    "Command being timed": "command",
    "User time (seconds)": "usertime",
    "System time (seconds)": "systemtime",
    "Percent of CPU this job got": "cpu",
    "Elapsed (wall clock) time (h:mm:ss or m:ss)": "elapsed",
    "Average shared text size (kbytes)": "avgtext",
    "Average unshared data size (kbytes)": "avgdata",
    "Average stack size (kbytes)": "avgstack",
    "Average total size (kbytes)": "avgtotal",
    "Maximum resident set size (kbytes)": "maxrss",
    "Average resident set size (kbytes)": "avgrss",
    "Major (requiring I/O) page faults": "major_pagefaults",
    "Minor (reclaiming a frame) page faults": "minor_pagefaults",
    "Voluntary context switches": "voluntaryswitches",
    "Involuntary context switches": "involuntaryswitches",
    "Swaps": "swaps",
    "File system inputs": "fileinputs",
    "File system outputs": "fileoutputs",
    "Socket messages sent": "socketsend",
    "Socket messages received": "socketrecv",
    "Signals delivered": "signals",
    "Page size (bytes)": "page_size",
    "Exit status": "status",
}


GNU_TIME_DEFAULT_VAL_TYPE = {
    "int": [
        "avgtext",
        "avgdata",
        "avgstack",
        "avgtotal",
        "maxrss",
        "avgrss",
        "major_pagefaults",
        "minor_pagefaults",
        "voluntaryswitches",
        "involuntaryswitches",
        "swaps",
        "fileinputs",
        "fileoutputs",
        "socketsend",
        "socketrecv",
        "signals",
        "page_size",
        "status",
        "usersystemtime",
    ],
    "float": [
        "usertime",
        "systemtime",
    ],
    "timedelta": [
        "elapsed",
    ],
    "percent": [
        "cpu",
    ],
}


GNU_TIME = True
cmd = "/usr/bin/time -v"
returncode, out, err, stats = run(cmd, logfd=None, gnu_time=GNU_TIME)
if returncode != 0:
    GNU_TIME = False


class DataDefinition(StrEnum):
    ENCAPP_DEFINITION = "encapp_definition"
    ENCAPP_RESULT = "encapp_result"
    OUTPUT_FILE = "output_file"  # file(s) produced by the test


def initialize_testdata():
    return {
        DataDefinition.ENCAPP_DEFINITION: None,
        DataDefinition.ENCAPP_RESULT: None,
        DataDefinition.OUTPUT_FILE: None,
    }


def encapp_is_installed(android_serial: str, debug: int):
    # check whether it is already installed
    already_installed = encapp_tool.app_utils.install_ok(android_serial, debug)
    if already_installed:
        return
    encapp_tool.app_utils.install_app(android_serial, debug)


def list_codecs(ava_config):
    output_dict = {
        "testname": "list_codecs",
    }

    try:
        # 0. preparation
        # 0.1. ensure encapp is installed
        encapp_is_installed(ava_config.android_serial, ava_config.debug)

        # 1. run encapp command
        model = "model"
        # use default workdir
        device_workdir = encapp.get_workdir(ava_config.android_serial)
        outfile = encapp.list_codecs(
            ava_config.android_serial,
            model,
            device_workdir=device_workdir,
            debug=ava_config.debug,
        )
        # 2. read and clean up the output file (json)
        codec_list_dict = encapp.read_json_file(outfile, ava_config.debug)
        pathlib.Path.unlink(outfile)
        # 3. write output
        output_dict["retcode"] = 0
        output_dict["results"] = codec_list_dict

    except Exception as e:
        output_dict["retcode"] = -1
        output_dict["error"] = repr(e)
    return output_dict


def list_codecs_for_serial(serial: str, device_workdir: str = None, debug: int = 0):
    try:
        # 0. preparation
        # 0.1. ensure encapp is installed
        encapp_is_installed(serial, debug)

        # 1. run encapp command
        model = "model"
        # use default workdir
        if not device_workdir:
            device_workdir = encapp.get_workdir(serial)
        outfile = encapp.list_codecs(
            serial,
            model,
            device_workdir=device_workdir,
            debug=debug,
        )
        # 2. read and clean up the output file (json)
        codec_list_dict = encapp.read_json_file(outfile, debug)
        pathlib.Path.unlink(outfile)
        # 3. write output
        return codec_list_dict

    except Exception as e:
        # TODO: what to do here?
        print(f"Codec lookup failed: {e}")

    return {}


def get_device_workdir():
    return "/sdcard"


def get_local_workdir(local_workdir: str):
    if local_workdir is None:
        # choose a random dir
        tempdir = tempfile.gettempdir()
        local_workdir = tempfile.mkdtemp(prefix="ava.tmp.", dir=tempdir)
    # prepare the local working directory to pull the files in
    if not os.path.exists(local_workdir):
        os.mkdir(local_workdir)
    return local_workdir


def get_android_serial(android_serial: str):
    if android_serial is not None:
        return android_serial
    elif "ANDROID_SERIAL" in os.environ:
        # read serial number from ANDROID_SERIAL env variable
        return os.environ["ANDROID_SERIAL"]
    else:
        return None


def get_all_serials(specific_serial: str = ""):
    if not specific_serial:
        specific_serial = ""

    status, std, stderr, _ = run("adb devices")
    devices = []

    text = std.decode()
    specific_serials = []
    if specific_serial != "":
        if "," in specific_serial:
            specific_serials = specific_serial.split("'")
        else:
            specific_serials = [specific_serial]

    for line in text.split("\n"):
        if "List of devices attached" in line:
            continue
        if "device" in line:
            serial = line.split("\t")[0]
            if specific_serial != "" and (serial not in specific_serials):
                continue
            devices.append(serial)
    return devices


def gnu_time_parse(gnu_time_str: str, logfd: int, debug: int):
    gnu_time_stats = {}
    for line in gnu_time_str.split("\n"):
        if not line:
            # empty line
            continue
        # check if we know the line
        line = line.strip()
        for key1, key2 in GNU_TIME_DEFAULT_KEY_DICT.items():
            if line.startswith(key1):
                break
        else:
            # unknown key
            print(f"warn: unknown gnutime line: {line}", file=logfd)
            continue
        val = line[len(key1) + 1 :].strip()
        # fix val type
        if key2 in GNU_TIME_DEFAULT_VAL_TYPE["int"]:
            val = int(val)
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["float"]:
            val = float(val)
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["percent"]:
            val = float(val[:-1])
        elif key2 in GNU_TIME_DEFAULT_VAL_TYPE["timedelta"]:
            # '0:00.02'
            timedelta_re = r"((?P<min>\d+):(?P<sec>\d+).(?P<centisec>\d+))"
            res = re.search(timedelta_re, val)
            timedelta_sec = int(res.group("min")) * 60 + int(res.group("sec"))
            timedelta_centisec = int(res.group("centisec"))
            timedelta_sec += timedelta_centisec / 100.0
            val = timedelta_sec
        gnu_time_stats[key2] = val
    gnu_time_stats["usersystemtime"] = (
        gnu_time_stats["usertime"] + gnu_time_stats["systemtime"]
    )
    return gnu_time_stats


def encapp_get_encoder_name(output_dict: dict[str, Any], mime_type: str):
    filtered_canonical_names = [
        encoder["canonical_name"]
        for encoder in output_dict["results"]["encoders"]
        if encoder.get("is_encoder", True)
        and encoder.get("is_hardware_accelerated", True)
        and encoder.get("media_type", {}).get("mime_type") == mime_type
    ]
    return filtered_canonical_names


def setup_test_for_input_file(
    test: encapp.tests_definitions.Test,
    input_file: str,
    device: list[str],
    mediastore: str,
    files_to_push: list[str],
) -> None:
    videoinfo = encapp.encapp_tool.ffutils.get_video_info(input_file)
    device_workdir = device.get("device_workdir", "")
    serial = device.get("serial", "")
    videoname = pathlib.Path(input_file).stem

    # Set basic video properties - encapp will handle transcoding
    test.input.framerate = int(round(float(videoinfo["framerate"]), 0))
    test.input.resolution = f"{videoinfo['width']}x{videoinfo['height']}"
    # Use host path - encapp will handle device file management
    test.input.filepath = input_file
    test.common.id = f"{serial}.{videoname}"


def run_capture(
    device: dict[str, Any],
    input_file: str,
    local_setup: dict[str, str],
    quality_param: str,
    bitrate_mode: str,
    test_name: str,
    merge: encapp.tests_definitions.Test = None,
) -> tuple[str, dict[str, Any]]:
    """
    Simplify runnning a single encoding
    Quality param can be a quality level or a bitrate
    A test can be provided that will be merged.
    """

    testdata = initialize_testdata()

    test_suite = encapp.tests_definitions.TestSuite()
    test = encapp.tests_definitions.Test()

    # Looks at the source file and sets props accordingly. We need unique files.
    setup_test_for_input_file(
        test,
        input_file,
        device,
        local_setup["mediastore"],
        files_to_push,
    )

    # Add the actual test name to make it truly unique
    # TODO:...
    test.common.description = ""  # f"{__doc__}"

    # Try to run as fast as posible
    test.input.realtime = False

    # Actual test defintion starts here
    test.configure.codec = device["encoder"]
    # TODO: what should be used in the general case?
    if bitrate_mode == "vbr":
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.vbr
        test.configure.bitrate = quality_param
    elif bitrate_mode == "cbr":
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.cbr
        test.configure.bitrate = quality_param
    if bitrate_mode == "cq":
        test.configure.bitrate_mode = encapp.tests_definitions.Configure.BitrateMode.cq
        test.configure.quality = quality_param

    test.common.id = f"{test_name}.{test.common.id}.{bitrate_mode}.{quality_param}"
    test.common.output_filename = test.common.id

    # Do the merge
    if merge:
        test.MergeFrom(merge)

    # Add to test suite
    test_suite.test.extend([test])
    # Write the test to file since it will have an anonymized name otherwise
    pbtxt_file = f"{local_setup['local_workdir']}/{test.common.id}.pbtxt"
    encapp.configfile_write(test_suite, pbtxt_file)
    testdata[DataDefinition.ENCAPP_DEFINITION] = pathlib.Path(pbtxt_file).name
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
    assert returncode, "error: encapp failed"

    # We have a result to work with
    output_files = result[1]

    # VERIFY result
    assert len(output_files) > 0, (
        f"Wrong number of outputs from the test: {len(output_files)}"
    )
    return output_files, testdata


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


def is_vbr_supported(device, encoder: str) -> bool:
    encoder_info = device.get("encoder_info", None)
    assert encoder_info
    mt = encoder_info.get("media_type", None)
    assert mt, "No mediatype"
    caps = mt.get("encoder_capabilities", None)
    assert caps, "No caps found for the codec"

    vbr_support = bool(caps.get("MODE_VBR"))
    return vbr_support


def is_cbr_supported(device, encoder: str) -> bool:
    encoder_info = device.get("encoder_info", None)
    assert encoder_info
    mt = encoder_info.get("media_type", None)
    assert mt, "No mediatype"
    caps = mt.get("encoder_capabilities", None)
    assert caps, "No caps found for the codec"

    cbr_support = bool(caps.get("MODE_CBR"))
    return cbr_support


def is_cq_supported(device, encoder: str) -> bool:
    encoder_info = device.get("encoder_info", None)
    assert encoder_info
    mt = encoder_info.get("media_type", None)
    assert mt, "No mediatype"
    caps = mt.get("encoder_capabilities", None)
    assert caps, "No caps found for the codec"

    cq_support = bool(caps.get("MODE_CQ"))
    return cq_support
