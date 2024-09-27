#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 22-06-2024 15:08:25

from pathlib import Path
from abc import abstractmethod
from typing import Protocol, Union, Iterable

from typing_extensions import Self
from seriallib import SerialProtocol

from .types import Comm


def is_simple(_type) -> bool:
    return _type in (str, int, float, bool, type(None))


def is_iterable(obj):
    return issubclass(type(obj), Iterable)


class SettingsProtocol(Protocol):
    lmpname: str
    delta_safe: int


class StateMgrProtocol(SerialProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback): ...

    @abstractmethod
    def first_run(self) -> None: ...

    @abstractmethod
    def restart(self, endflag: bool, restartfile: Union[Path, None], ptr: Union[int, None]) -> None: ...

    @abstractmethod
    def attach(self, comm: Comm, max_time: int) -> None: ...


class NoTimeLeft(RuntimeError):
    pass
