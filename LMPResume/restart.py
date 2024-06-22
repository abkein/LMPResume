#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 16-06-2024 21:36:13

import sys
import time
import shlex
import argparse
import subprocess
from pathlib import Path


def restart(time: int, scriptpath: Path):
    cmd = f"LMPResumeRestart {shlex.quote(scriptpath.as_posix())} --time={time}"
    cmds = shlex.split(cmd)
    subprocess.Popen(cmds, start_new_session=True)


def main():
    parser = argparse.ArgumentParser('LMPResume')
    parser.add_argument("module")
    parser.add_argument("--time", action="store", type=int, required=True)
    args = parser.parse_args()

    cmd = f"LMPResume run {shlex.quote(Path(args.module).as_posix())}"
    cmds = shlex.split(cmd)

    time.sleep(args.time)

    subprocess.Popen(cmds, start_new_session=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
