#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import json
import toml
import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Union, Any, Type

# from mpi4py import MPI  // imported as needed
import pysbatch_ng
from pysbatch_ng.execs import CMD
from indexlib import Index

from .state import StateManager, StateManagerSchema
from .meta import NoTimeLeft, FirstRunFallbackTrigger


filename_conffile: str = "sbatch.toml"


def load_script(module_path: Path) -> ModuleType:
    module_name = module_path.parts[-1]

    module_init = module_path / "__init__.py"
    if not module_init.exists():
        with module_init.open("w") as fp:
            fp.write("# This file was automatically created by LMPResume")
            fp.write("")
            fp.write(f"from . import {module_name}")

    spec = importlib.util.spec_from_file_location(module_name, module_init.as_posix(), submodule_search_locations=[module_path.as_posix()])

    if spec is None: raise ImportError(f"Cannot import module by path {module_path.as_posix()}: spec is None")
    if spec.loader is None: raise ImportError(f"Cannot import module by path {module_path.as_posix()}: spec.loader is None")

    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    script: ModuleType = getattr(module, module_name)
    return script


def load_schema(script: ModuleType) -> Type[StateManagerSchema]:
    schema_name = "ManagerSchema"
    if not hasattr(script, schema_name):
        raise RuntimeError(f"Script has no '{schema_name}' object")
    schema: Type[StateManagerSchema] | None = getattr(script, schema_name, None)
    if schema is None:
        raise RuntimeError(f"'{schema_name}' object is None")
    elif not issubclass(schema, StateManagerSchema):
        raise TypeError(f"Specified '{schema_name}' object is not a subclass of 'StateManagerSchema'")

    return schema


class AZAZ:
    simulation: StateManager
    cwd: Path
    managerSchema: StateManagerSchema
    endflag: bool
    internal: bool
    restartfile: Path | None = None
    conffile: Path
    modulepath: Path
    tag: int | None = None
    __norestart: bool = False
    valgrind: bool
    valgrind_track_origin: bool

    def __init__(self):
        parser = argparse.ArgumentParser('LMPResume')
        parser.add_argument("module", action="store", type=str, help="Module path")
        parser.add_argument("--internal", action="store_true", default=False)
        parser.add_argument("--valgrind", action="store_true", default=False)
        parser.add_argument("--valgrind_track_origin", action="store_true", default=False)
        parser.add_argument('--cwd', action='store', type=str, default=None, help="Current working directory. If not specified, default unix pwd is used.")
        parser.add_argument('--max_time', action='store', type=int, default=0, help="Max run time in seconds. If unknown, default 0 should be used. -1 if unlimited.")

        parser.add_argument('--capture', choices=['yes', 'no', 'release'], type=str, default='release')

        parser.add_argument('--tag', action='store', type=int, default=None)
        parser.add_argument('--conf', action='store', type=str, default=None)

        parser.add_argument('--end', action='store_true')
        parser.add_argument('--norestart', action='store_true')
        parser.add_argument('--restartfile', action='store', type=str, default=None, help="Restart file to read.")
        parser.add_argument('--ptr', action='store', type=int, default=0, help="Default: 0")
        args = parser.parse_args()

        self.internal = bool(args.internal)

        self.cwd = Path.cwd()
        if args.cwd is not None:
            self.cwd = Path(args.cwd).resolve()
            if not self.cwd.exists():
                raise FileNotFoundError(f"Specified cwd does not exists: {self.cwd.as_posix()}")

        self.conffile = Path(args.conf).resolve() if args.conf is not None else self.cwd / filename_conffile

        self.valgrind = args.valgrind
        self.valgrind_track_origin = args.valgrind_track_origin

        data: dict[str, Any] = {"cwd": self.cwd.as_posix()}

        if not self.restart_flag:
            do_capture: Union[bool, None] = None
            if args.capture == 'no':        do_capture = False
            elif args.capture == 'yes':     do_capture = True
            elif args.capture == 'release': do_capture = None

            if args.capture is not None:
                data["capture"] = do_capture

        self.modulepath = Path(args.module).resolve()
        if not self.modulepath.exists():
            raise FileNotFoundError(f"Specified module path does not exists: {self.modulepath.as_posix()}")

        self.managerSchema = load_schema(load_script(self.modulepath))()

        self.endflag = bool(args.end)

        max_time: int = int(args.max_time)

        data["max_time"] = max_time

        if args.tag is not None:
            self.tag = args.tag

        if args.restartfile is not None:
            self.restartfile = Path(args.restartfile).resolve()
            if not self.restartfile.exists():
                raise FileNotFoundError(f"Specified restart file does not exists: {self.restartfile.as_posix()}")

        ptr: Union[int, None] = None
        if args.ptr is not None:
            ptr = int(args.ptr)
            if ptr < 0:
                raise RuntimeError("Specified ptr cannot be less than zero")

        if ptr is not None:
            data["ptr"] = ptr

        if self.restart_flag:
            with self.statefile.open('r') as fp:
                d = json.load(fp)
            d.update(data)
            data = d

        if args.norestart is not None:
            self.__norestart = True

        _sim = self.managerSchema.load(data)
        assert isinstance(_sim, StateManager)
        self.simulation = _sim

    def dumpit(self):
        d = self.managerSchema.dump(self.simulation)
        assert isinstance(d, dict)
        with self.statefile.open('w') as fp:
            json.dump(d, fp)

    def get_sbatch(self):
        pysbatch_ng.log.configure('screen')
        if self.conffile.exists():
            with self.conffile.open('r') as fp:
                d = toml.load(fp)
            sbatch = pysbatch_ng.Sbatch.from_schema(d)
            if not isinstance(sbatch, pysbatch_ng.Sbatch):
                raise RuntimeError("")
            if not sbatch.check(False):
                raise RuntimeError("")
        else:
            sbatch = pysbatch_ng.Sbatch(pysbatch_ng.Options(), pysbatch_ng.Platform(), self.cwd)
        return sbatch

    def make_index(self):
        index = Index(self.cwd)
        found, _, cat = index.find_category("slurm")
        if not found:
            index.register_category("slurm", "Slurm related files")
        if not index.isregistered(self.conffile):
            index.register(self.conffile, "slurm", False, "Main slurm configuration file")
        if index.issub(self.modulepath):
            found, _, cat = index.find_category("sim")
            if not found:
                index.register_category("sim", "Simulation related files")
            if not index.isregistered(self.modulepath):
                index.register(self.modulepath, "sim", True, "Simulation run file")
        index.commit()

    def reborn(self) -> int:
        self.make_index()
        self.simulation.run_no += 1
        self.dumpit()

        sbatch = self.get_sbatch()

        sbatch.options.tag = self.tag
        sbatch.options.job_number = self.simulation.run_no
        if not sbatch.check(False):
            return 1

        max_time = sbatch.platform.get_timelimit(sbatch.options.partition) if sbatch.options.partition is not None else 0
        _exec = "LMPResume"
        args_cmd = f"{self.modulepath.as_posix()} --internal --cwd={self.cwd.as_posix()} --max_time={max_time}"
        if self.valgrind:
            _exec = "valgrind"
            args_cmd = f" LMPResume " + args_cmd
            if self.valgrind_track_origin:
                args_cmd = " --track-origins=yes" + args_cmd
        if self.endflag:
            args_cmd += f" --end"

        if sbatch.options.cmd is None:
            sbatch.options.cmd = CMD(
                executable=_exec,
                args=args_cmd
            )
        else:
            sbatch.options.cmd.executable = _exec
            sbatch.options.cmd.args = args_cmd

        after_args = f"{self.modulepath.as_posix()} --cwd={self.cwd.as_posix()} --conf={self.conffile.as_posix()}"
        if self.tag is not None:
            after_args += f" --tag={self.tag}"
        after_cmd = CMD(
            executable="LMPResume",
            args=after_args
        )

        sbatch.run(True, poll_cmd=after_cmd)
        return 0

    def main(self) -> int:
        from mpi4py import MPI
        self.simulation.attach(self.dumpit, MPI.COMM_WORLD)
        try:
            with self.simulation as st:
                if self.restart_flag:
                    try:
                        st.restart(self.endflag, self.restartfile)
                    except FirstRunFallbackTrigger:
                        st.first_run()
                else:
                    st.first_run(self.__norestart)
        except NoTimeLeft:
            pass
        finally:
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.dumpit()

        MPI.COMM_WORLD.Barrier()

        MPI.Finalize()
        return 0

    def run(self) -> int:
        return self.main() if self.internal else self.reborn()

    @property
    def statefile(self) -> Path:
        return self.cwd / "state.json"

    @property
    def restart_flag(self) -> bool:
        return self.statefile.exists()


def main() -> int:
    return AZAZ().run()


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.exit(main())
