#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 21-10-2024 04:41:54

import time
import logging
from pathlib import Path
from abc import abstractmethod
from typing import Any, Iterable, Callable, Union, Type
from typing_extensions import Self

import lammps
import lammps.formats
from marshmallow import Schema, fields

from .util import CaptureManager, minilog, FieldPath
from .meta import NoTimeLeft, FirstRunFallbackTrigger, StateMgrProtocol, Comm


class StateManagerSchema(Schema):
    cwd = FieldPath(missing=Path.cwd())
    run_no = fields.Integer()
    ptr = fields.Integer()
    max_time = fields.Integer(missing=0, load_only=True)
    lmpname = fields.String(missing="mpi")
    delta_safe = fields.Integer(missing=5*60)
    do_capture = fields.Boolean(allow_none=True, data_key='capture')
    state = fields.Dict(keys=fields.String())


class StateManager(StateMgrProtocol):
    ptr: int
    cwd: Path
    rank: int
    size: int
    comm: Comm
    run_no: int
    lmpname: str
    delta_safe: int
    max_time: float
    starttime: float
    capturefile: Path
    lmp: lammps.lammps
    restart_folder: Path
    state: dict[str, Any]
    logger: logging.Logger
    do_capture: bool | None
    capture: CaptureManager

    @abstractmethod
    def user_init(self) -> None: ...

    def init(self) -> None:
        self.user_init()
        self.capturefile = self.cwd / "capturedump"
        self.restart_folder = self.cwd / "restarts"
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.logger = minilog("LMPResume").getChild(f"{self.rank}")
        self.rootlog(f"The world is {self.size}")
        self.starttime = time.time()
        self.capture = CaptureManager(self.capturefile, self.do_capture)

    def __init__(
        self,
        do_capture: bool | None = None,
        cwd: Path = Path.cwd(),
        max_time: int = 0,
        lmpname: str = "mpi",
        delta_safe: int = 5*60,
        run_no: int = -1,
        ptr: int = 0,
        state: dict[str, Any] = {}
        ) -> None:
        self.cwd = cwd
        self.do_capture = do_capture
        self.max_time = max_time
        self.lmpname = lmpname
        self.delta_safe = delta_safe
        self.run_no = run_no
        self.ptr = ptr
        self.state = state

    def attach(self, comm: Comm) -> None:
        self.comm = comm

        self.init()

    def rootlog(self, message: str) -> None:
        if self.rank == 0: self.logger.info(message)

    def __enter__(self) -> Self:
        self.lmp = lammps.lammps(self.lmpname)

        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception, exc_traceback) -> None:
        self.shutdown()

        if self.rank == 0 and exc_value is not None and exc_type != NoTimeLeft: self.logger.exception(exc_value)

        self.lmp.close()
        # self.lmp.finalize()

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
        a_restart = (self.restart_folder / 'restart.a')
        b_restart = (self.restart_folder / 'restart.b')

        if a_restart.exists():
            with self.capture.file():
                lmp_1 = lammps.lammps(self.lmpname)
                lmp_1.command(f"read_restart {a_restart.as_posix()}")
                lmp_1.command(f"run 0")
                step_a: Union[int, None] = lmp_1.get_thermo("step")
                lmp_1.close()

            if step_a is None: raise RuntimeError("Error getting step")
        else:
            step_a = -1

        if b_restart.exists():
            with self.capture.file():
                lmp_2 = lammps.lammps(self.lmpname)
                lmp_2.command(f"read_restart {b_restart.as_posix()}")
                lmp_2.command(f"run 0")
                step_b: Union[int, None] = lmp_2.get_thermo("step")
                lmp_2.close()

            if step_b is None: raise RuntimeError("Error getting step")
        else:
            step_b = -1

        if step_a == -1 and step_b == -1:
            raise FirstRunFallbackTrigger()

        if step_a > step_b:
            return self.restart_folder / 'restart.a'
        else:
            return self.restart_folder / 'restart.b'

    def read_restart(self, restartfile: Path) -> None:
        self.logger.debug(f"Reading restart {restartfile.as_posix()}")
        cmd = f"read_restart {restartfile.relative_to(self.cwd).as_posix()}"
        self.logger.debug(f"CMD is: '{cmd}'")
        with self.capture.file():
            self.lmp.command(cmd)
            self.lmp.command(f"run 0")
        self.logger.debug("Readed restart")

    def first_run(self) -> None:
        self.logger.info("Running first time")

        self.build()

        self.startup()

        self.initialize_system()

        with self.capture.file(): self.lmp.command(f"run 0")

        self.setup()

        self.run(False)

    def restart(self, endflag: bool, restartfile: Union[Path, None]) -> None:
        self.startup()

        _restartfile: Path
        if restartfile is not None:
            _restartfile = restartfile
        else:
            _restartfile = self.find_restart()

        self.logger.info(f"Using restart {_restartfile.as_posix()}")

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


if __name__ == "__main__":
    pass

