#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pathlib import Path
from abc import abstractmethod
from typing import Protocol, Union, Iterable, Callable


from .types import Comm


def is_simple(_type) -> bool:
    return _type in (str, int, float, bool, type(None))


def is_iterable(obj):
    return issubclass(type(obj), Iterable)


class SettingsProtocol(Protocol):
    lmpname: str
    delta_safe: int


class StateMgrProtocol:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def __enter__(self): ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback): ...

    @abstractmethod
    def first_run(self) -> None: ...

    @abstractmethod
    def restart(self, endflag: bool, restartfile: Union[Path, None]) -> None: ...  # , ptr: Union[int, None]) -> None: ...

    @abstractmethod
    def attach(self, dump_callback: Callable[[], None], comm: Comm) -> None: ...


class NoTimeLeft(RuntimeError):
    pass


class FirstRunFallbackTrigger(RuntimeError):
    pass


if __name__ == "__main__":
    pass
