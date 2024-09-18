#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 25-06-2024 07:00:41

import sys
import json
import time
import logging
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Generator, Tuple, List, Callable, Union, Type

import lammps
import lammps.formats
from typing_extensions import Self

from .util import CaptureManager
from .serial import MakeDecoder, UniversalJSONEncoder
from .meta import NoTimeLeft, SettingsProtocol, StateProxyProtocol, StateMgrProtocol, Comm


class StateProxy(StateProxyProtocol):
    __mgr: StateMgrProtocol

    def __init__(self, mgr: StateMgrProtocol, *args, **kwargs) -> None:
        self.__mgr = mgr
        # for k, v in kwargs.items():
        #     self.__store[k] = v

    # def __enter__(self) -> Self:
    #     self.__lmp = lammps.lammps(self.lmpname)
    #     return self

    # def __exit__(self, exc_type, exc_value, exc_traceback):
    #     self.dump()
    #     self.__lmp.finalize()

    def rebind(self) -> Self:
        return self.__mgr.rebind()  # type: ignore

    def has_time(self) -> bool:
        return self.remaining_time > 0

    def check_time(self) -> bool:
        if not self.has_time(): raise NoTimeLeft()
        else: return True

    @property
    def remaining_time(self) -> float:
        return self.__mgr.settings.max_time - (time.time() - self.__mgr.starttime) - self.__mgr.settings.delta_safe

    @property
    def lmp(self) -> lammps.lammps:
        return self.__mgr.lmp

    @property
    def run_no(self) -> int:
        return self.__mgr.run_no

    @property
    def comm(self) -> Comm:
        return self.__mgr.comm

    @property
    def capture(self) -> CaptureManager:
        return self.__mgr.capture

    def update(self, _dict: Dict[Any, Any]) -> Self:
        for k, v in _dict.items(): self[self._keytransform(k)] = v
        return self

    def items(self) -> Generator[Tuple[str, Any], Any, None]:
        for k, v in self.__mgr.state.items():
            yield k, v

    def __repr__(self) -> str:
        return self.__mgr.state.__repr__()

    def __contains__(self, key: Any) -> bool:
        return self._keytransform(key) in self.__mgr.state

    def __getitem__(self, key: Any) -> Any:
        return self.__mgr.state[self._keytransform(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.__mgr.state[self._keytransform(key)] = value

    def __delitem__(self, key: Any) -> None:
        del self.__mgr.state[self._keytransform(key)]

    def __iter__(self) -> Generator[str, Any, None]:
        # self.ptr = 0
        for el in self.__mgr.state:
            yield el
        # return self

    # def __next__(self) -> str:
        # if self.ptr > len(self):
        #     raise StopIteration
        # self.ptr += 1
        # return self[self._store.keys()[self.ptr-1]]

    def _keytransform(self, key: Any) -> str:
        return str(key)

    def __len__(self) -> int:
        return len(self.__mgr.state)


@dataclass
class Settings(SettingsProtocol):
    startup: Callable[[StateProxy], None]
    build: Callable[[StateProxy], None]
    scheme: List[Callable[[StateProxy], None]]  # type: ignore
    setup: Callable[[StateProxy], None]
    restart: Callable[[StateProxy], None]
    end: Callable[[StateProxy], None]
    shutdown: Callable[[StateProxy], None]
    max_time: int
    lmpname: str
    types: Union[List[Type], None]
    delta_safe: int


def minilog(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter: logging.Formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
    soutHandler = logging.StreamHandler(stream=sys.stdout)
    soutHandler.setLevel(logging.DEBUG)
    soutHandler.setFormatter(formatter)
    logger.addHandler(soutHandler)
    serrHandler = logging.StreamHandler(stream=sys.stderr)
    serrHandler.setFormatter(formatter)
    serrHandler.setLevel(logging.WARNING)
    logger.addHandler(serrHandler)
    return logger


class StateManager(StateMgrProtocol):
    cwd: Path
    run_no: int
    ptr: int
    state: Dict[str, Any]
    lmp: lammps.lammps
    statefile: Path
    starttime: float
    settings: Settings
    scriptpath: Path
    thermofile: Path
    is_restart: bool
    comm: Comm
    rank: int
    size: int
    logger: logging.Logger
    capturefile: Path
    capture: CaptureManager

    def __init__(self, scriptpath: Path, do_capture: Union[bool, None], cwd: Path, comm: Comm, *args, **kwargs) -> None:
        self.cwd = cwd
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.logger = minilog("LMPResume").getChild(f"{self.rank}")
        self.logger.info(f"The world is {self.size}")
        self.run_no = 0
        self.starttime = time.time()
        self.statefile = Path.cwd() / ".state.json"
        self.thermofile = Path.cwd() / "thermo.csv"
        self.scriptpath = scriptpath
        self.capturefile = Path.cwd() / "capturedump"
        self.capture = CaptureManager(self.capturefile, do_capture)
        self.ptr = 0
        self.state = {}
        self.load_script(self.scriptpath)
        self.is_restart = self.statefile.exists()
        if self.is_restart:
            self.rootlog("LMPResume: Using configuration in statefile")
            self.load()

    def rootlog(self, message: str) -> None:
        if self.rank == 0: self.logger.info(message)

    def __json__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "__type": type(self).__name__,
            "run_no": self.run_no,
            "ptr": self.ptr,
            "state": self.state,
        }
        return d

    @classmethod
    def __rejs__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)
        instance.check_2type_set("run_no", int, **kwargs)
        instance.check_2type_set("ptr", int, **kwargs)
        instance.check_2type_set("state", dict, **kwargs)
        return instance

    def __enter__(self) -> Self:
        self.lmp = lammps.lammps(self.settings.lmpname)

        return self

    def rebind(self) -> StateProxy:
        self.dump()
        self.lmp.close()
        self.lmp = lammps.lammps(self.settings.lmpname)
        return self.proxy

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, exc_traceback) -> None:
        self.settings.shutdown(self.proxy)

        self.dump()
        self.lmp.close()
        self.lmp.finalize()

        if exc_type != NoTimeLeft and self.rank == 0:
            (self.cwd / "NORESTART").touch(exist_ok=True)

    def dump(self) -> None:
        if self.rank == 0:
            text = json.dumps(self, cls=UniversalJSONEncoder, indent=4)
            with self.statefile.open('w') as fp:
                fp.write(self._obfuscate(text))

    def _obfuscate(self, text: str) -> str:
        return text

    def _deobfuscate(self, text: str) -> str:
        return text

    def load(self) -> None:
        with self.statefile.open('r') as fp:
            text = self._deobfuscate(fp.read())
        tlst = [StateManager]
        if self.settings.types is not None: tlst += self.settings.types
        obj: StateManager = json.loads(text, cls=MakeDecoder(tlst))
        self.ptr = obj.ptr
        self.state = obj.state
        self.run_no = obj.run_no + 1

    def load_script(self, module_path: Path) -> None:
        module_init = module_path / "__init__.py"
        module_name = module_path.parts[-1]
        if not module_init.exists():
            with module_init.open("w") as fp:
                fp.write(
                      """
                            # This file was automatically created by LMPResume

                            from . import script

                        """
                )

        spec = importlib.util.spec_from_file_location(module_name, module_init.as_posix(), submodule_search_locations=[module_path.as_posix()])

        if spec is None: raise ImportError(f"Cannot import module by path {module_path.as_posix()}\nSomething went wrong")
        elif spec.loader is None: raise ImportError(f"Cannot import module by path {module_path.as_posix()}\nSomething went wrong")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        script = getattr(module, module_name)
        self.settings = script.LMPResume_setting

    @property
    def proxy(self) -> StateProxy:
        return StateProxy(self)

    def run(self, endflag: bool = False) -> None:
        if self.run_no == 0:
            with self.capture.file(): self.settings.build(self.proxy)

        self.logger.debug(f"Endflag is {'true' if endflag else 'false'}")
        self.settings.startup(self.proxy)
        if self.is_restart:
            self.logger.info("Running restart")
            with self.capture.file(): self.settings.restart(self.proxy)


        self.logger.info("Setting up")
        with self.capture.file(): self.settings.setup(self.proxy)
        if self.is_restart:
            with self.capture.file(): self.lmp.command("run 0")

        if not endflag:
            last_ptr = self.ptr
            for i, func in enumerate(self.settings.scheme):
                if i < last_ptr: continue
                self.logger.info(f"Running {func.__name__}")
                with self.capture.file(): func(self.proxy)
                self.ptr += 1
                if time.time() - self.starttime > self.settings.max_time - self.settings.delta_safe:
                    raise NoTimeLeft()

        with self.capture.file(): self.settings.end(self.proxy)
