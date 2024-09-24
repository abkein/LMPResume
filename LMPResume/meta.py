#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 22-06-2024 15:08:25

from pathlib import Path
from inspect import isfunction
from abc import abstractmethod
from typing import Tuple, Type, Protocol, Dict, Any, Union, Iterable, get_origin, get_args, runtime_checkable

from typing_extensions import Self

from .types import Comm


def is_simple(_type) -> bool:
    return _type in (str, int, float, bool, type(None))


def is_default_constructible(_type) -> bool:
    return _type in (dict, list, str, int, float, bool, type(None), Path)


def is_iterable(obj):
    return issubclass(type(obj), Iterable)


@runtime_checkable
class SerialProtocol(Protocol):
    def check_check_set(self, arg: str, _type: Tuple[Type, ...], **kwargs) -> None:
        if arg not in kwargs: raise RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' not in kwargs")
        obj = kwargs[arg]
        if not isinstance(obj, _type):
            raise RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' is not of type '{_type}'")
        self.__setattr__(arg, obj)

    def check_2type_set(self, arg: str, to_type: Type, accept_none: bool = False, **kwargs) -> None:
        if arg not in kwargs: raise RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' not in kwargs")
        obj = kwargs[arg]
        if isinstance(obj, type(None)) and not accept_none:
            raise RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' is None")
        self.__setattr__(arg, to_type(obj))

    def __json__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "__type": type(self).__name__
        }
        for k, v in self.__dict__.items():
            if not (k.startswith("__") or k.startswith(f"_{type(self).__name__}")):
                d[k] = v
        return d

    @classmethod
    def __rejs__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)
        for k, v in cls.__annotations__.items():
            if k.startswith("__") or k.startswith(f"_{cls.__name__}"): continue
            __v_origin = get_origin(v)
            v_origin: Type = v if __v_origin is None else __v_origin
            if is_default_constructible(v_origin): instance.check_2type_set(k, v_origin, False, **kwargs)
            elif v_origin == Union:
                _types = get_args(v)
                if len(_types) == 2:
                    __types = list(_types)
                    fl = type(None) in __types
                    if fl: del __types[__types.index(type(None))]
                    if is_default_constructible(__types[0]): instance.check_2type_set(k, __types[0], fl, **kwargs)
                else: instance.check_check_set(k, _types, **kwargs)
            else:
                print(v)
                instance.check_check_set(k, v, **kwargs)

        return instance

    def __hash__(self) -> int:
        return hash(self.__dict__)


def has_method(_type, method: str) -> bool:
    if hasattr(_type, method):
        _method = getattr(_type, method, None)
        if _method is not None:
            return isfunction(_method)

    return False


def isserializable(_type) -> bool:
    return has_method(_type, "__json__") and has_method(_type, "__rejs__") and has_method(_type, "__hash__")


class SettingsProtocol(Protocol):
    lmpname: str
    delta_safe: int


class StateMgrProtocol(SerialProtocol):
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
