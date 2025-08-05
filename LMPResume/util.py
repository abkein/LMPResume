#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import io
import sys
import logging
import tempfile
from pathlib import Path
from typing import Type, Union

from marshmallow import fields


class FieldPath(fields.Field):
    def _deserialize(self, value: str, attr, data, **kwargs) -> Path:
        return Path(value).resolve(True)

    def _serialize(self, value: Path, attr, obj, **kwargs) -> str:
        return value.as_posix()


def minilog(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter: logging.Formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
    sout_handler = logging.StreamHandler(stream=sys.stdout)
    sout_handler.setLevel(logging.DEBUG)
    sout_handler.setFormatter(formatter)
    logger.addHandler(sout_handler)
    serr_handler = logging.StreamHandler(stream=sys.stderr)
    serr_handler.setFormatter(formatter)
    serr_handler.setLevel(logging.WARNING)
    logger.addHandler(serr_handler)
    return logger


class OutputCaptureStr:  # derived from lammps.OutputCapture
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

    def __enter__(self):
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

    def __enter__(self):
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


if __name__ == "__main__":
    pass
