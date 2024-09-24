#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 10-08-2024 12:14:56

from typing import Union

from mpi4py import MPI

Comm = Union[MPI.Intercomm, MPI.Intracomm]
