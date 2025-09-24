#!/usr/bin/env python3

import argparse
import importlib
import os
import pandas as pd
import pathlib
import sys
import tempfile
import traceback
import google
from google.protobuf import text_format
import json

ava_common = importlib.import_module("ava-common")

# allow importing from subprojects
root_path = os.path.join(os.path.dirname(__file__), "..")
lib_path = os.path.join(root_path, "lib", "encapp", "scripts")
sys.path.insert(0, lib_path)

import encapp
import encapp_tool

ENCODED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi")
# TODO: figure out how to handle this. On osx the /usr/bin/time is not the GNU version.

GNU_TIME = True
cmd = "/usr/bin/time -v"
returncode, out, err, stats = ava_common.run(cmd, logfd=None, gnu_time=GNU_TIME)
if returncode != 0:
    GNU_TIME = False


def encapp_is_installed(android_serial, debug):
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

