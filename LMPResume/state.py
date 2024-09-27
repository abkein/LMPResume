#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 25-09-2024 12:29:30

import time
import logging
from pathlib import Path
from abc import abstractmethod
from typing import Dict, Any, Iterable, Callable, Union, Type
from typing_extensions import Self

import lammps
import lammps.formats
from seriallib import SerialProtocol

from .util import CaptureManager, minilog
from .meta import NoTimeLeft, StateMgrProtocol, Comm


class StateManager(StateMgrProtocol):
    cwd: Path  # saved
    run_no: int # saved
    ptr: int # saved
    state: Dict[str, Any] # saved
    lmp: lammps.lammps # runtime set
    starttime: float # runtime set
    max_time: float # runtime set
    lmpname: str # saved
    delta_safe: int # saved
    comm: Comm # runtime set
    rank: int # runtime set
    size: int # runtime set
    logger: logging.Logger # runtime set
    capturefile: Path # saved
    do_capture: Union[bool, None] # saved
    capture: CaptureManager # runtime set
    restartPath: Path # saved

    @abstractmethod
    def user_init(self) -> None: ...

    def init(self) -> None:
        self.user_init()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = minilog("LMPResume").getChild(f"{self.rank}")
        self.rootlog(f"The world is {self.size}")
        self.starttime = time.time()
        self.capture = CaptureManager(self.capturefile, self.do_capture)

    def __init__(self, do_capture: Union[bool, None], cwd: Path, comm: Comm, max_time: int, lmpname: str = "mpi", delta_safe: int = 5*60, *args, **kwargs) -> None:
        self.cwd = cwd
        self.comm = comm
        self.capturefile = self.cwd / "capturedump"
        self.restartPath = self.cwd / "restarts"
        self.do_capture = do_capture
        self.max_time = max_time
        self.lmpname = lmpname
        self.delta_safe = delta_safe
        self.run_no = 0
        self.ptr = 0
        self.state = {}

        self.init()

    def attach(self, comm: Comm, max_time: int = 0) -> None:
        self.comm = comm
        self.max_time = max_time

        self.init()

    def rootlog(self, message: str) -> None:
        if self.rank == 0: self.logger.info(message)

    def __2dict__(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "__type": type(self).__name__,
            "run_no": self.run_no,
            "ptr": self.ptr,
            "cwd": self.cwd.as_posix(),
            "lmpname": self.lmpname,
            "state": self.state,
            "delta_safe": self.delta_safe,
            "do_capture": self.do_capture,
            "restartPath": self.restartPath,
            "capturefile": self.capturefile,
        }
        return d

    @classmethod
    def __4dict__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)
        instance.check_2type_set("run_no", int, **kwargs)
        instance.check_2type_set("ptr", int, **kwargs)
        instance.check_2type_set("cwd", Path, **kwargs)
        instance.check_2type_set("restartPath", Path, **kwargs)
        instance.check_2type_set("capturefile", Path, **kwargs)
        instance.check_2type_set("delta_safe", int, **kwargs)
        instance.check_2type_set("lmpname", str, **kwargs)
        instance.check_2type_set("state", dict, **kwargs)
        instance.check_2type_set("do_capture", bool, True, **kwargs)

        return instance

    def __enter__(self) -> Self:
        self.lmp = lammps.lammps(self.lmpname)

        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, exc_traceback) -> None:
        self.shutdown()

        if self.rank == 0 and exc_value is not None: self.logger.exception(exc_value)

        self.lmp.close()
        self.lmp.finalize()

        if exc_type != NoTimeLeft and self.rank == 0:
            (self.cwd / "NORESTART").touch(exist_ok=True)

    def rebind(self) -> lammps.lammps:
        self.lmp.close()
        self.lmp = lammps.lammps(self.lmpname)
        return self.lmp

    @abstractmethod
    def scheme(self) -> Iterable[Callable[[], None]]: ...

    @abstractmethod
    def build(self): ...

    @abstractmethod
    def startup(self): ...

    @abstractmethod
    def initialize_system(self): ...

    @abstractmethod
    def setup(self): ...

    @abstractmethod
    def shutdown(self): ...

    @abstractmethod
    def end(self): ...

    def time_check(self) -> None:
        if time.time() - self.starttime > self.max_time - self.delta_safe:
            raise NoTimeLeft()

    def find_restart(self) -> Path:
        with self.capture.file():
            lmp_1 = lammps.lammps(self.lmpname)
            lmp_1.command(f"read_restart {(self.restartPath / 'restart.a').as_posix()}")
            lmp_1.command(f"run 0")
            step_a: Union[int, None] = lmp_1.get_thermo("step")
            lmp_1.close()
            lmp_1.finalize()
        if step_a is None: raise RuntimeError("Error getting step")

        with self.capture.file():
            lmp_2 = lammps.lammps(self.lmpname)
            lmp_2.command(f"read_restart {(self.restartPath / 'restart.a').as_posix()}")
            lmp_2.command(f"run 0")
            lmp_2.close()
            lmp_2.finalize()
            step_b: Union[int, None] = lmp_2.get_thermo("step")
        if step_b is None: raise RuntimeError("Error getting step")

        if step_a > step_b:
            return self.restartPath / 'restart.a'
        else:
            return self.restartPath / 'restart.b'

    def read_restart(self, restartfile: Path) -> None:
        with self.capture.file():
            self.lmp.command(f"read_restart {restartfile.as_posix()}")
            self.lmp.command(f"run 0")

    def first_run(self) -> None:
        self.build()

        self.startup()

        self.initialize_system()

        with self.capture.file(): self.lmp.command(f"run 0")

        self.setup()

        self.run(False)

    def restart(self, endflag: bool, restartfile: Union[Path, None], ptr: Union[int, None]) -> None:
        if ptr is not None:
            self.ptr = ptr

        self.startup()

        _restartfile: Path
        if restartfile is not None:
            _restartfile = restartfile
        else:
            _restartfile = self.find_restart()

        self.read_restart(_restartfile)

        with self.capture.file(): self.lmp.command(f"run 0")

        self.setup()

        self.run(endflag)


    def run(self, endflag: bool) -> None:
        if not endflag:
            last_ptr = self.ptr
            for i, func in enumerate(self.scheme()):
                if i < last_ptr: continue

                self.logger.info(f"Running {func.__name__}")
                with self.capture.file(): func()

                self.ptr += 1
                self.time_check()

        with self.capture.file(): self.end()


class LoopStageProtocol(SerialProtocol):
    stage_key: str
    stage_len: int
