#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 20-06-2024 19:16:56

import os
import sys
import argparse
from typing import Union
from pathlib import Path

from mpi4py import MPI

from .meta import NoTimeLeft
from .state import StateManager


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser('LMPResume')
    parser.add_argument("module", action="store", type=str)
    parser.add_argument('--cwd', action='store', type=str, default=None)
    parser.add_argument('--capture', choices=['yes', 'no', 'release'], type=str, default='release')
    parser.add_argument('--end', action='store_true')

    cwd = Path.cwd()

    # sub_parsers = parser.add_subparsers(help="sub-command help", dest="command")

    # parser_init = sub_parsers.add_parser("run", help="Run simulation")
    # parser_init.add_argument("module")

    # parser_thermo = sub_parsers.add_parser("thermo", help="Extract thermo from file")
    # parser_thermo.add_argument("file")

    args = parser.parse_args()

    if args.cwd is not None: os.chdir(Path(args.cwd).resolve())

    do_capture: Union[bool, None] = None
    if args.capture == 'no':    do_capture = False
    elif args.capture == 'yes': do_capture = True
    else:                       do_capture = None

    try:
        with StateManager(Path(args.module).resolve(), do_capture, cwd, MPI.COMM_WORLD) as st:
            st.run(args.end)
            pass
    except NoTimeLeft as e:
        pass

    # MPI.Finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
