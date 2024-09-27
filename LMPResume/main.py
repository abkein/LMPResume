#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 20-06-2024 19:16:56

import os
import sys
import json
import argparse
import importlib.util
from types import ModuleType
from typing import Union, Tuple, Type, Any, Iterable
from pathlib import Path

from mpi4py import MPI
from seriallib import MakeDecoder, UniversalJSONEncoder, isserializable

from .meta import NoTimeLeft, StateMgrProtocol
from .state import StateManager


def deobfuscate(s: str) -> str:
    return s


def obfuscate(s: str) -> str:
    return s


def load(statefile: Path, gen_type: Type[StateMgrProtocol], types: Union[None, Iterable[Type[Any]]]) -> StateMgrProtocol:
    with statefile.open('r') as fp:
        text = deobfuscate(fp.read())
    tlst = [StateManager]
    if types is not None: tlst += types
    return gen_type.__4dict__(**json.loads(text, cls=MakeDecoder(tlst)))


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


def parse_args() -> Tuple[Path,  Union[bool, None], Path, bool, int, Union[Path, None], Union[int, None]]:
    parser = argparse.ArgumentParser('LMPResume')
    parser.add_argument("module", action="store", type=str, help="Module path")
    parser.add_argument('--cwd', action='store', type=str, default=None, help="Current working directory. If not specified, default unix pwd is used.")
    parser.add_argument('--max_time', action='store', type=int, default=0, help="Max run time in seconds. If unknown, default 0 should be used. -1 if unlimited.")
    parser.add_argument('--restartfile', action='store', type=str, default=None, help="Restart file to read.")
    parser.add_argument('--ptr', action='store', type=int, default=0, help="Default: 0")
    parser.add_argument('--capture', choices=['yes', 'no', 'release'], type=str, default='release')
    parser.add_argument('--end', action='store_true')
    args = parser.parse_args()

    if args.cwd is not None:
        newcwd = Path(args.cwd).resolve()
        if not newcwd.exists():
            raise FileNotFoundError(f"Specified cwd does not exists: {newcwd.as_posix()}")
        os.chdir(newcwd)

    cwd = Path.cwd()

    do_capture: Union[bool, None] = None
    if args.capture == 'no':    do_capture = False
    elif args.capture == 'yes': do_capture = True
    else:                       do_capture = None

    scriptpath = Path(args.module).resolve()
    if not scriptpath.exists():
        raise FileNotFoundError(f"Specified module path does not exists: {scriptpath.as_posix()}")

    endflag: bool = bool(args.end)

    max_time: int = int(args.max_time)

    restf: Union[Path, None] = None
    if args.restartfile is not None:
        restf = Path(args.restartfile).resolve()
        if not restf.exists():
            raise FileNotFoundError(f"Specified restart file does not exists: {restf.resolve()}")

    ptr: Union[int, None] = None
    if args.ptr is not None:
        ptr = int(args.ptr)
        if ptr < 0:
            raise RuntimeError("Specified ptr cannot be less than zero")

    return cwd, do_capture, scriptpath, endflag, max_time, restf, ptr


def load_types(script: ModuleType) -> Tuple[Union[None, Tuple[Type[Any], ...]], Type[StateMgrProtocol]]:
    types_tuple: Union[None, Tuple[Type[Any], ...]] = None

    if not hasattr(script, "simulation"):
        raise RuntimeError("Script has no 'simulation' object")
    _sim: Union[Type[StateMgrProtocol], None] = getattr(script, "simulation", None)
    if _sim is None:
        raise RuntimeError("'simulation' object is None")
    elif not isinstance(_sim, type):
        raise TypeError(f"Specified 'simulation' object is not a class: {str(type(_sim))}")
    elif not issubclass(_sim, StateMgrProtocol):
        raise TypeError("Specified 'simulation' class is not inherited from 'StateMgrProtocol'")

    if hasattr(script, "types2serialize"):
        types_tuple = getattr(script, "types2serialize", None)
    if types_tuple is not None:
        if not isinstance(types_tuple, tuple):
            raise RuntimeError("Loaded types tuple is not a Tuple")
        else:
            for _type in types_tuple:
                if not isserializable(_type):
                    raise RuntimeError(f"Type: {str(_type)} is not serializable")

    return types_tuple, _sim


def dump(file: Path, simulation: StateMgrProtocol) -> None:
    if MPI.COMM_WORLD.Get_rank() == 0:
        text = json.dumps(simulation, cls=UniversalJSONEncoder, indent=4)
        with file.open('w') as fp:
            fp.write(obfuscate(text))


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "1"

    cwd, do_capture, scriptpath, endflag, max_time, restf, ptr = parse_args()

    script = load_script(scriptpath)

    types_tuple, simType = load_types(script)

    statefile = cwd / "state.json"
    simulation: StateMgrProtocol
    restart_flag = statefile.exists()
    if restart_flag:
        simulation = load(statefile, simType, types_tuple)
        simulation.attach(MPI.COMM_WORLD, max_time)
    else:
        simulation = simType(do_capture, cwd, MPI.COMM_WORLD, max_time)

    try:
        with simulation as st:
            if restart_flag:
                st.restart(endflag, restf, ptr)
            else:
                st.first_run()
    except NoTimeLeft:
        MPI.COMM_WORLD.Barrier()
    finally:
        dump(statefile, simulation)

    # MPI.Finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
