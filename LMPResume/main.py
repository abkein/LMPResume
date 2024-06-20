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
from pathlib import Path

from .meta import NoTimeLeft
from .state import StateManager


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser('LMPResume')

    sub_parsers = parser.add_subparsers(help="sub-command help", dest="command")

    parser_init = sub_parsers.add_parser("run", help="Run simulation")
    parser_init.add_argument("module")

    parser_thermo = sub_parsers.add_parser("thermo", help="Extract thermo from file")
    parser_thermo.add_argument("file")

    args = parser.parse_args()

    if args.command == "run":
        try:
            with StateManager(Path(args.module).resolve()) as st:
                st.run()
                pass
        except NoTimeLeft as e:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
