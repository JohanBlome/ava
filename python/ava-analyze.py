import json
import importlib
import os.path
from pathlib import Path
import sys
import datetime
from pprint import pprint
import re
import argparse

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

# noqa: E402:
import encapp


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input', help='', default=None)
    parser.add_argument('-o','--output', help='', default=None)

    options = parser.parse_args(sys.argv[1:])
    testresult = None
    with open(options.input, "r") as f:
        print("Load json")
        testresult = json.load(f)
    pprint(testresult)




if __name__ == "__main__":
    main()
