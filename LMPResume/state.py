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
    dump_callback: Callable[[], None]

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

    def attach(self, dump_callback: Callable[[], None], comm: Comm) -> None:
        self.comm = comm
        def fake_callback():
            return None

        self.dump_callback = dump_callback if comm.Get_rank() == 0 else fake_callback

        self.init()

    def rootlog(self, message: str) -> None:
        if self.rank == 0: self.logger.info(message)

    def __enter__(self) -> Self:
        self.lmp = lammps.lammps(self.lmpname, comm=self.comm)

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
        self.lmp = lammps.lammps(self.lmpname, comm=self.comm)
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
            self.dump_callback()
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



class Stage:
    stage_key: str
    stage_len: int
    manager: StateManager
    additional: int = 0

    def __init__(self, manager: StateManager, additional: int = 0, name: str | None = None) -> None:
        self.manager = manager
        self.additional = additional
        if name is not None:
            self.stage_key = name
        if not hasattr(self, "stage_key"):
            self.stage_key = self.__class__.__name__

    @abstractmethod
    def go(self): ...


class LongStage(Stage):
    def get_time_tries(self) -> tuple[float, int]:
        total_time: float = 0
        tries: int = 0
        if self.stage_key in self.manager.state:
            total_time = self.manager.state[self.stage_key]["total_time"]
            tries      = self.manager.state[self.stage_key]["tries"]
        else:
            self.manager.state[self.stage_key] = {
                "total_time": total_time,
                "tries": tries
                }
        return total_time, tries

    def go(self):
        total_time, tries = self.get_time_tries()
        self.manager.dump_callback()
        print(f"Starting {self.stage_key}")

        self.prerun()
        for it in range(tries, self.stage_len):
            start_time = time.time()

            self.run(it)

            end_time = time.time()
            total_time += end_time - start_time
            tries += 1
            self.manager.state[self.stage_key] = {
                "total_time": total_time,
                "tries": tries
                }

            self.manager.time_check()

        self.manager.dump_callback()
        self.postrun()
        self.manager.dump_callback()
        print(f"{self.stage_key} end")

    @abstractmethod
    def prerun(self): ...

    @abstractmethod
    def run(self, it: int): ...

    @abstractmethod
    def postrun(self): ...


class FastStage(Stage):
    def go(self):
        if self.stage_key in self.manager.state:
            started = self.manager.state[self.stage_key]["started"]
            ended      = self.manager.state[self.stage_key]["ended"]
            if started and (not ended):
                self.manager.logger.warning(f"The stage {self.stage_key} has been started, but not ended. Starting over...")
        else:
            self.manager.state[self.stage_key] = {
                "started": True,
                "ended": False
                }
        self.manager.dump_callback()
        print(f"Starting {self.stage_key}")
        self.run()
        self.manager.state[self.stage_key]["ended"] = True
        self.manager.dump_callback()
        print(f"{self.stage_key} end")

    @abstractmethod
    def run(self): ...


if __name__ == "__main__":
    pass

