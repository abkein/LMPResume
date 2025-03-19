#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import argparse



def main():
    parser_main = argparse.ArgumentParser('LMPMGR', description="LMPResume job manager", add_help=True)

    subs_main = parser_main.add_subparsers(dest="command")

    ### LIST
    parser_list = subs_main.add_parser('list', help='List active jobs')


if __name__ == "__main__":
    sys.exit(main())
