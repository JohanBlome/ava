import os
import importlib
from pprint import pprint
import re
import pandas as pd
from typing import Any
import sys
import ava_common


def get_qpextract() -> dict[str, Any]:
    qpextract_cmd = "qpextract"
    resources = {"cmd": "qpextract", "version": ""}
    # check if it s available on in the path
    # qpextract -h
    # qpextract  v1.0.15
    returncode, out, stderr, stats = ava_common.run(f"{qpextract_cmd} -h")
    print(f"{out}")
    if not returncode:
        m = re.search(r"(?P<version>[0-9]*\.[0-9]*\.[0-9]*)", str(out))
        if m:
            resources["version"]
    return resources


def extract_simple_stream(mediafile: str, outputfile: str) -> int:
    """Convert a mp4 to annexb using ffmpeg"""
    print(f"Extract simple stream: {mediafile} -> {outputfile}")
    ffmpeg_command = (
        f"ffmpeg -i {mediafile} -vcodec copy -bsf hevc_mp4toannexb {outputfile}"
    )
    returncode, out, err, stats = ava_common.run(ffmpeg_command)

    print(f"{returncode=} {err=}")
    return returncode


def get_qpbounds(qpcsv: str) -> tuple[int, int]:
    qpd = pd.read_csv(qpcsv)
    qp_min = int(qpd["qp_min"].min())
    qp_max = int(qpd["qp_max"].max())
    return qp_min, qp_max


def get_qpstats(h265_file: str, cmd: str = "", chroma: bool = False) -> dict[str, int]:
    assert cmd is not None, "tool is None"
    data = {}
    modes = ["y", "cb", "cr"]
    cargs = {"y": "qpymode", "cb": "qpcbmode", "cr": "qpcrmode"}

    if cmd == "":
        cmd = get_qpextract()["cmd"]
    if chroma:
        for mode in modes:
            # ?
            carg = cargs[mode]
            qpfile = f"{h265_file}.qp.{mode}.csv"
            run_cmd = f"{cmd} --{carg} -i {h265_file} -o {qpfile}  "
            print(f"{run_cmd=}")
            returncode, out, err, stats = ava_common.run(run_cmd)
            assert returncode == 0, f"error: {out = } {err = }"
            qp_min, qp_max = get_qpbounds(qpfile)
            data[f"qpmin_{mode}"] = qp_min
            data[f"qpmax_{mode}"] = qp_max
            pprint(data)
            assert returncode == 0, f"error: {out = } {err = }"
    else:
        qpfile = f"{h265_file}.qp.csv"
        if not os.path.exists(qpfile):
            run_cmd = f"{cmd} -i {h265_file} -o {qpfile}  "
            returncode, out, err, stats = ava_common.run(run_cmd)
            assert returncode == 0, f"error: {out = } {err = }"
            qp_min, qp_max = get_qpbounds(qpfile)
            data["qpmin"] = qp_min
            data["qpmax"] = qp_max
        else:
            print(f"File {qpfile} already exists")

    return data
