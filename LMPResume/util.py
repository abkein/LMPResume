#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 25-06-2024 07:00:38

import os
import io
import sys
import tempfile
from pathlib import Path
from typing import Type, Union, Dict

from typing_extensions import Self


# derived from lammps.OutputCapture
class OutputCaptureStr:
    """ Utility class to capture library output into string variable """

    stdout_fd: int
    captured_output: str
    tmpfile: io.BufferedRandom
    stdout_orig: int
    do_capture: bool

    def __init__(self, do_capture: bool = True) -> None:
        self.do_capture = do_capture
        self.stdout_fd = sys.stdout.fileno()
        self.captured_output = ""

    def __enter__(self) -> Self:
        if self.do_capture:
            self.tmpfile = tempfile.TemporaryFile(mode='w+b')

            sys.stdout.flush()

            # make copy of original stdout
            self.stdout_orig = os.dup(self.stdout_fd)

            # replace stdout and redirect to temp file
            os.dup2(self.tmpfile.fileno(), self.stdout_fd)
        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, traceback) -> None:
        if self.do_capture:
            os.dup2(self.stdout_orig, self.stdout_fd)
            os.close(self.stdout_orig)
            self.tmpfile.close()

    @property
    def output(self) -> str:
        if self.do_capture:
            sys.stdout.flush()
            self.tmpfile.flush()
            self.tmpfile.seek(0, io.SEEK_SET)
            self.captured_output = self.tmpfile.read().decode('utf-8')
        return self.captured_output


# derived from lammps.OutputCapture
class OutputCaptureFile:
    """ Utility class to capture library output into file """

    stdout_fd: int
    stdout_orig: int
    do_capture: bool
    file: Path
    fp: io.TextIOWrapper

    def __init__(self, file: Path, do_capture: bool = True) -> None:
        self.file = file
        self.do_capture = do_capture
        self.stdout_fd = sys.stdout.fileno()

    def __enter__(self) -> Self:
        if self.do_capture:
            self.fp = self.file.open('a')

            sys.stdout.flush()

            # make copy of original stdout
            self.stdout_orig = os.dup(self.stdout_fd)

            # replace stdout and redirect to temp file
            os.dup2(self.fp.fileno(), self.stdout_fd)
        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, traceback) -> None:
        if self.do_capture:
            sys.stdout.flush()
            self.fp.flush()
            self.fp.seek(0, io.SEEK_SET)
            os.dup2(self.stdout_orig, self.stdout_fd)
            os.close(self.stdout_orig)
            self.fp.close()


class CaptureManager:
    __do_capture: Union[bool, None]
    __file: Path

    def __init__(self, file: Path, do_capture: Union[bool, None] = None) -> None:
        self.__do_capture = do_capture
        self.__file = file.resolve()

    def file(self, file: Union[Path, None] = None, do_capture: Union[bool, None] = None) -> OutputCaptureFile:
        cap: bool = True
        if self.__do_capture is not None: cap = self.__do_capture
        elif do_capture is not None: cap = do_capture
        return OutputCaptureFile(file if file is not None else self.__file, cap)

    def string(self, do_capture: Union[bool, None] = None) -> OutputCaptureStr:
        cap: bool = True
        if self.__do_capture is not None: cap = self.__do_capture
        elif do_capture is not None: cap = do_capture
        return OutputCaptureStr(cap)
