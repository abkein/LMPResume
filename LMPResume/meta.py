#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 22-06-2024 15:08:25

from pathlib import Path
from abc import abstractmethod
from typing import Generator, List, Callable, Tuple, Type, Protocol, Dict, Any, Union, Iterable, get_origin, get_args, runtime_checkable

from mpi4py import MPI
import lammps
from typing_extensions import Self

from .util import CaptureManager


Comm = Union[MPI.Intercomm, MPI.Intracomm]


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
            RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' is not of type '{_type}'")
        self.__setattr__(arg, obj)

    def check_2type_set(self, arg: str, to_type: Type, accept_none: bool = False, **kwargs) -> None:
        if arg not in kwargs: raise RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' not in kwargs")
        obj = kwargs[arg]
        if isinstance(obj, type(None)):
            if not accept_none:
                raise RuntimeError(f"Cannot create '{type(self).__name__}' instance: '{arg}' is None")
        self.__setattr__(arg, to_type(obj))

    def __json__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "__type": type(self).__name__
            # "__serialized": "__json__"
        }
        for k, v in self.__dict__.items():
            if not (k.startswith("__") or k.startswith(f"_{type(self).__name__}")):
                # if isinstance(v, SerialProtocol):
                #     d[k] = v.__json__()
                # else: d[k] = v
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


class StateProxyProtocol(Protocol):
    __mgr: Any

    @abstractmethod
    def rebind(self): ...

    @property
    @abstractmethod
    def remaining_time(self) -> float: ...

    @property
    @abstractmethod
    def lmp(self) -> lammps.lammps: ...

    @property
    @abstractmethod
    def run_no(self) -> int: ...

    @abstractmethod
    def update(self, _dict: Dict[str, Any]) -> Self: ...

    @abstractmethod
    def items(self) -> Generator[Tuple[str, Any], Any, None]: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __contains__(self, key: int) -> bool: ...

    @abstractmethod
    def __getitem__(self, key: str) -> Any: ...

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def __delitem__(self, key: str) -> None: ...

    @abstractmethod
    def __iter__(self) -> Generator[str, Any, None]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class SettingsProtocol(Protocol):
    scheme: List[Callable[[StateProxyProtocol], None]]
    setup: Callable[[StateProxyProtocol], None]
    restart: Callable[[StateProxyProtocol], None]
    max_time: int
    lmpname: str
    types: Union[List[Type], None]
    delta_safe: int


class StateMgrProtocol(SerialProtocol):
    run_no: int
    ptr: int
    state: Dict[str, Any]
    lmp: lammps.lammps
    statefile: Path
    starttime: float
    settings: SettingsProtocol
    scriptpath: Path
    comm: Comm
    capture: CaptureManager

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def rebind(self) -> StateProxyProtocol: ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback): ...

    @abstractmethod
    def dump(self) -> None: ...

    @abstractmethod
    def _obfuscate(self, text: str) -> str: ...

    @abstractmethod
    def _deobfuscate(self, text: str) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def load_script(self, module_path: Path) -> None: ...

    @property
    @abstractmethod
    def proxy(self) -> StateProxyProtocol: ...

    @abstractmethod
    def run(self) -> None: ...


class NoTimeLeft(RuntimeError):
    pass
