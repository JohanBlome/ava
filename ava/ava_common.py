#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import tempfile
import pathlib
from typing import Any, List
from enum import StrEnum
from google.protobuf import text_format

# Add lib/encapp/scripts to Python path for encapp imports
root_path = os.path.join(os.path.dirname(__file__), "..")
lib_path = os.path.join(root_path, "lib", "encapp", "scripts")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Import encapp modules
try:
    import encapp
    import encapp_tool
except ImportError as e:
    print(f"Warning: Could not import encapp modules: {e}")
    encapp = None
    encapp_tool = None

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
    if not encapp_tool:
        raise ImportError("encapp_tool not available")
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
    if not encapp:
        print("encapp not available")
        return {}
        
    try:
        # 0. preparation
        # 0.1. ensure encapp is installed
        encapp_is_installed(serial, debug)

        # 1. run encapp command with timeout
        model = "model"
        # use default workdir
        if not device_workdir:
            device_workdir = encapp.get_workdir(serial)
            if not device_workdir:
                device_workdir = "/sdcard"
        
        print(f"Getting codecs for device {serial} (this may take a moment)...")
        outfile = encapp.list_codecs(
            serial,
            model,
            device_workdir=device_workdir,
            debug=debug,
        )
        # 2. read and clean up the output file (json)
        codec_list_dict = encapp.read_json_file(outfile, debug)
        pathlib.Path.unlink(outfile)
        
        # 3. Process the codec list to extract encoders with proper structure
        if 'encoders' in codec_list_dict:
            processed_encoders = []
            for encoder in codec_list_dict['encoders']:
                # Extract mime_type from media_type if it exists
                mime_type = encoder.get('mime_type', '')
                if 'media_type' in encoder and isinstance(encoder['media_type'], dict):
                    mime_type = encoder['media_type'].get('mime_type', mime_type)
                
                # Create a processed encoder with mime_type at top level
                processed_encoder = encoder.copy()
                processed_encoder['mime_type'] = mime_type
                processed_encoders.append(processed_encoder)
            
            codec_list_dict['encoders'] = processed_encoders
        
        # 4. write output
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



def get_compatible_pix_fmt(device: dict, encoder: str) -> Any:
    """Get a compatible pixel format for the encoder based on its capabilities."""
    if not encapp:
        return encapp.tests_definitions.PixFmt.nv12  # Default fallback
    
    try:
        encoder_info = device.get("encoder_info", None)
        if not encoder_info:
            return encapp.tests_definitions.PixFmt.nv12  # Default fallback
            
        mt = encoder_info.get("media_type", None)
        if not mt:
            return encapp.tests_definitions.PixFmt.nv12  # Default fallback
            
        caps = mt.get("encoder_capabilities", None)
        if not caps:
            return encapp.tests_definitions.PixFmt.nv12  # Default fallback
        
        # Check for supported color formats in order of preference
        # Priority: nv12 (most common), yuv420p, nv21, rgba
        color_formats = caps.get("color_formats", [])
        
        # Map Android color format constants to encapp PixFmt
        color_format_map = {
            20: encapp.tests_definitions.PixFmt.nv12,  # COLOR_FormatYUV420SemiPlanar
            19: encapp.tests_definitions.PixFmt.yuv420p,  # COLOR_FormatYUV420Planar
            21: encapp.tests_definitions.PixFmt.nv21,  # COLOR_FormatYUV420PackedSemiPlanar
            22: encapp.tests_definitions.PixFmt.rgba,  # COLOR_Format32bitARGB8888
        }
        
        # Find the first supported format in order of preference
        for color_format in [20, 19, 21, 22]:  # nv12, yuv420p, nv21, rgba
            if color_format in color_formats:
                return color_format_map[color_format]
        
        # If no specific formats found, default to nv12
        return encapp.tests_definitions.PixFmt.nv12
        
    except Exception as e:
        print(f"Warning: Could not determine compatible pixel format: {e}")
        return encapp.tests_definitions.PixFmt.nv12  # Default fallback


def setup_test_for_mp4_input(
    test: Any,
    input_file: str,
    device: list[str],
    mediastore: str,
) -> None:
    """Setup test for MP4 input with device decoding (transcoding)"""
    if not encapp or not encapp_tool:
        raise ImportError("encapp not available")
        
    videoinfo = encapp.encapp_tool.ffutils.get_video_info(input_file)
    device_workdir = device.get("device_workdir", "")
    serial = device.get("serial", "")
    encoder = device.get("encoder", "")
    
    # For MP4 transcoding, we push the original file and let the device decode it
    videoname = pathlib.Path(input_file).stem
    mp4file = f"{videoname}.mp4"
    
    # Configure for MP4 input with device decoding
    # Use a compatible pixel format based on encoder capabilities
    test.input.pix_fmt = get_compatible_pix_fmt(device, encoder)
    test.input.framerate = int(round(float(videoinfo["framerate"]), 0))
    test.input.resolution = f"{videoinfo['width']}x{videoinfo['height']}"
    # Use full path - encapp will copy to mediastore and update paths automatically
    test.input.filepath = input_file
    test.common.id = f"{serial}.{videoname}"


def setup_test_for_input_file(
    test: Any,
    input_file: str,
    device: list[str],
    mediastore: str,
) -> None:
    print(f"DEBUG: setup_test_for_input_file called with input_file: {input_file}")
    if not encapp or not encapp_tool:
        print("DEBUG: encapp not available in setup_test_for_input_file")
        raise ImportError("encapp not available")
        
    print("DEBUG: Getting video info")
    videoinfo = encapp.encapp_tool.ffutils.get_video_info(input_file)
    device_workdir = device.get("device_workdir", "")
    serial = device.get("serial", "")
    print(f"DEBUG: Video info: {videoinfo}")

    # Set basic video properties - encapp will handle transcoding
    test.input.framerate = int(round(float(videoinfo["framerate"]), 0))
    test.input.resolution = f"{videoinfo['width']}x{videoinfo['height']}"
    # Use full path - encapp will copy to mediastore and update paths automatically
    test.input.filepath = input_file
    print(f"DEBUG: setup_test_for_input_file complete, filepath: {test.input.filepath}")


# OLD run_capture function removed - replaced with create_test_template() and run_encapp_test()


# Define pixelrates and bitrates in kbps
pixelrates = {1920 * 1080 * 30: 8000, 1920 * 1080 * 60: 10000, 1280 * 720 * 30: 6000}


def find_bitrate_for_video(videopath: str, encoder: str) -> str:
    if not encapp or not encapp_tool:
        # Fallback bitrate calculation without encapp
        return "1000kbps"
        
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


def run_encapp_stats_to_csv(json_files: List[str], output_dir: str) -> List[str]:
    """Run encapp_stats_to_csv.py to generate performance CSV files from JSON files"""
    if not encapp or not encapp_tool:
        print("Warning: encapp not available, cannot generate performance CSV files")
        return []
    
    csv_files = []
    
    try:
        # Look for encapp_stats_to_csv.py in the lib/encapp directory
        encapp_stats_path = None
        possible_paths = [
            "lib/encapp/scripts/encapp_stats_to_csv.py",
            "../lib/encapp/scripts/encapp_stats_to_csv.py",
            "encapp_stats_to_csv.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                encapp_stats_path = path
                break
        
        if not encapp_stats_path:
            print("Warning: encapp_stats_to_csv.py not found, cannot generate performance CSV files")
            return []
        
        # Run encapp_stats_to_csv on all JSON files
        cmd = ["python3", encapp_stats_path] + json_files
        print(f"Running encapp_stats_to_csv: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find generated CSV files
        for json_file in json_files:
            csv_file = f"{json_file}_encoding_data.csv"
            if os.path.exists(csv_file):
                csv_files.append(csv_file)
                print(f"Generated performance CSV: {csv_file}")
        
    except Exception as e:
        print(f"Error running encapp_stats_to_csv: {e}")
    
    return csv_files


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


def create_test_template() -> Any:
    """
    Create a basic test template that tests can configure.
    AVA only provides the template - tests configure their own settings.
    """
    if not encapp:
        raise ImportError("encapp not available")
    
    test = encapp.tests_definitions.Test()
    
    # Set basic common properties
    test.common.description = ""
    test.input.realtime = False  # Run as fast as possible
    
    return test


def run_encapp_test(
    test: Any,
    device: dict[str, Any],
    input_file: str,
    local_setup: dict[str, str],
    test_name: str,
) -> tuple[list[str], dict[str, Any]]:
    """
    Run an encapp test that has been configured by the calling test.
    AVA only executes - it doesn't configure the test.
    """
    if not encapp:
        raise ImportError("encapp not available")

    testdata = initialize_testdata()
    test_suite = encapp.tests_definitions.TestSuite()
    
    # Add the test to the suite
    test_suite.test.extend([test])
    
    # Write the test to file
    pbtxt_file = f"{local_setup['local_workdir']}/{test.common.id}.pbtxt"
    encapp.configfile_write(test_suite, pbtxt_file)
    testdata[DataDefinition.ENCAPP_DEFINITION] = pathlib.Path(pbtxt_file).name
    
    # Set the source file to the original input file (MP4) for quality assessment
    testdata["sourcefile"] = input_file
    
    # Run the test - let encapp handle all the file processing and transcoding
    print(f"** Running codec test {test.common.id}")
    result = encapp.run_codec_tests(
        test_suite,
        [],  # No files to push - encapp handles this
        "na",
        device["serial"],
        local_setup["mediastore"],
        local_setup["local_workdir"],
        ignore_results=False,
        fast_copy=True,
        split=False,
        debug=1,
    )

    returncode = result[0]
    if not returncode:
        raise RuntimeError("encapp test failed")

    # Return the output files and test data
    output_files = result[1]
    if not output_files:
        raise RuntimeError(f"No output files from test: {test.common.id}")
    
    return output_files, testdata


def run_encapp_cli(pbtxt_file, device, workdir, mediastore="_mediastore"):
    """Run encapp CLI and return the output files."""
    try:
        # Ensure the workdir and all parent directories exist before running encapp CLI
        os.makedirs(workdir, exist_ok=True)
        
        # Convert to absolute paths to avoid working directory issues
        abs_workdir = os.path.abspath(workdir)
        abs_pbtxt_file = os.path.abspath(pbtxt_file)
        
        # Build encapp CLI command using direct script path
        encapp_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "encapp", "scripts", "encapp.py")
        cmd = [
            "python3", encapp_script,
            "run",
            abs_pbtxt_file,
            "--serial", device["serial"],
            "--local-workdir", abs_workdir,
            "--mediastore", mediastore,
            "--fast-copy"
        ]
        
        # Only add --device-workdir if it's not empty
        if device.get("device_workdir"):
            cmd.extend(["--device-workdir", device["device_workdir"]])
        
        print(f"Running encapp CLI: {' '.join(cmd)}")
        print(f"Working directory: {abs_workdir}")
        print(f"Device workdir: {device['device_workdir']}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Run encapp CLI from the current working directory (not workdir)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"encapp exit code: {result.returncode}")
        if result.stdout:
            print(f"encapp stdout: {result.stdout}")
        if result.stderr:
            print(f"encapp stderr: {result.stderr}")
        
        if result.returncode == 0:
            # Find output files in the workdir
            output_files = []
            for file in os.listdir(workdir):
                if file.endswith(('.json', '.mp4')):
                    output_files.append(os.path.join(workdir, file))
            
            print(f"Found output files: {output_files}")
            return True, output_files
        else:
            print(f"encapp failed with exit code {result.returncode}")
            return False, []
            
    except Exception as e:
        print(f"Error running encapp CLI: {e}")
        return False, []


def run_encapp_stats_to_csv(output_files, workdir, max_parallel=4):
    """Run encapp_stats_to_csv CLI on all JSON files and return the CSV path."""
    try:
        # Find all JSON files in the output (encapp_stats_to_csv expects JSON files, not MP4)
        json_files = [f for f in output_files if f.endswith('.json')]
        
        if not json_files:
            print("No JSON files found for encoder statistics analysis")
            return None
        
        # Convert relative paths to absolute paths for JSON files
        json_files_abs = []
        for json_file in json_files:
            if not os.path.isabs(json_file):
                # If it's a relative path, make it absolute relative to current working directory
                json_files_abs.append(os.path.abspath(json_file))
            else:
                json_files_abs.append(json_file)
        
        # encapp_stats_to_csv generates output files in the same directory as input files
        # with names like: {json_filename}_encoding_data.csv, {json_filename}_decoding_data.csv, etc.
        # We'll run it from the workdir so output files are created there
        
        # Build encapp_stats_to_csv CLI command using direct script path
        encapp_stats_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "encapp", "scripts", "encapp_stats_to_csv.py")
        cmd = [
            "python3", encapp_stats_script
        ]
        
        # Add JSON files as positional arguments (these should be the JSON output files from capture)
        cmd.extend(json_files_abs)
        
        # Find the project root by looking for the lib/encapp directory
        project_root = None
        current_dir = os.path.abspath(".")
        while current_dir != "/":
            if os.path.exists(os.path.join(current_dir, "lib", "encapp")):
                project_root = current_dir
                break
            current_dir = os.path.dirname(current_dir)
        
        if not project_root:
            project_root = os.path.abspath(".")  # Fallback to current directory
        
        print(f"Running encapp_stats_to_csv CLI: {' '.join(cmd)}")
        print(f"Working directory: {workdir}")
        print(f"JSON files to analyze: {json_files_abs}")
        print(f"Project root: {project_root}")
        
        # Run encapp_stats_to_csv CLI from the project root directory
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        print(f"encapp_stats_to_csv exit code: {result.returncode}")
        if result.stdout:
            print(f"encapp_stats_to_csv stdout: {result.stdout}")
        if result.stderr:
            print(f"encapp_stats_to_csv stderr: {result.stderr}")
        
        if result.returncode == 0:
            # Check for generated CSV files in the workdir
            # Look for files with patterns: *_encoding_data.csv, *_decoding_data.csv, *_named_ts_timestamps.csv
            generated_files = []
            for file in os.listdir(workdir):
                if (file.endswith('_encoding_data.csv') or 
                    file.endswith('_decoding_data.csv') or 
                    file.endswith('_named_ts_timestamps.csv')):
                    generated_files.append(os.path.join(workdir, file))
            
            if generated_files:
                print(f"Encoder statistics analysis completed. Generated files: {generated_files}")
                return generated_files  # Return list of generated files
            else:
                print("Warning: No CSV files generated by encapp_stats_to_csv")
                return None
        else:
            print(f"encapp_stats_to_csv failed with exit code {result.returncode}")
            return None
            
    except Exception as e:
        print(f"Error running encapp_stats_to_csv CLI: {e}")
        return None


def run_encapp_quality(output_files, workdir, mediastore="_mediastore", max_parallel=4):
    """Run encapp_quality CLI on all JSON files and return the CSV path."""
    try:
        # Find all JSON files in the output (encapp_quality expects JSON files, not MP4)
        json_files = [f for f in output_files if f.endswith('.json')]
        
        if not json_files:
            print("No JSON files found for quality analysis")
            return None
        
        # Convert relative paths to absolute paths for JSON files
        json_files_abs = []
        for json_file in json_files:
            if not os.path.isabs(json_file):
                # If it's a relative path, make it absolute relative to current working directory
                json_files_abs.append(os.path.abspath(json_file))
            else:
                json_files_abs.append(json_file)
        
        # Save quality output directly to workdir with a suitable name
        csv_filename = "quality_analysis.csv"
        csv_path = os.path.abspath(os.path.join(workdir, csv_filename))
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Use the provided mediastore path
        mediastore_path = os.path.abspath(mediastore)
        
        if not os.path.exists(mediastore_path):
            print(f"Warning: Mediastore directory not found: {mediastore_path}")
            # Try to create it
            try:
                os.makedirs(mediastore_path, exist_ok=True)
                print(f"Created mediastore directory: {mediastore_path}")
            except Exception as e:
                print(f"Failed to create mediastore directory: {e}")
                mediastore_path = os.path.abspath(".")  # Fallback to current directory
        
        # Build encapp_quality CLI command using direct script path
        encapp_quality_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "encapp", "scripts", "encapp_quality.py")
        cmd = [
            "python3", encapp_quality_script,
            "--max-parallel", str(max_parallel),
            "--csv",
            "--output", csv_path,
            "--media", mediastore_path,
            "--header",
            "--keep-quality-files",
            "--ignore-timing", "true",
            "--siti"  # Add SI/TI complexity analysis
        ]
        
        # Add JSON files as positional arguments (these should be the JSON output files from capture)
        cmd.extend(json_files_abs)
        
        # Find the project root by looking for the lib/encapp directory
        project_root = None
        current_dir = os.path.abspath(".")
        while current_dir != "/":
            if os.path.exists(os.path.join(current_dir, "lib", "encapp")):
                project_root = current_dir
                break
            current_dir = os.path.dirname(current_dir)
        
        if not project_root:
            project_root = os.path.abspath(".")  # Fallback to current directory
        
        print(f"Running encapp_quality CLI: {' '.join(cmd)}")
        print(f"Working directory: {workdir}")
        print(f"JSON files to analyze: {json_files_abs}")
        print(f"Output CSV: {csv_path}")
        print(f"Mediastore directory: {mediastore_path}")
        print(f"Project root: {project_root}")
        
        # Run encapp_quality CLI from the project root directory
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        print(f"encapp_quality exit code: {result.returncode}")
        if result.stdout:
            print(f"encapp_quality stdout: {result.stdout}")
        if result.stderr:
            print(f"encapp_quality stderr: {result.stderr}")
        
        if result.returncode == 0:
            # Check if CSV file was created
            if os.path.exists(csv_path):
                print(f"Quality analysis completed: {csv_path}")
                return csv_path
            else:
                print("Warning: No CSV file generated by encapp_quality")
                return None
        else:
            print(f"encapp_quality failed with exit code {result.returncode}")
            return None
            
    except Exception as e:
        print(f"Error running encapp_quality CLI: {e}")
        return None


def debug_protobuf(description: str, test_suite: encapp.tests_definitions.TestSuite):
    print(f"{description}: {text_format.MessageToString(test_suite)}")