import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Union, Dict, Any

import toml
import pysbatch_ng
from pysbatch_ng.utils import confdict


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


def loadconf(conffile:Path, conf_format: str, logger: logging.Logger) -> Dict[str, Any]:
    if not conffile.exists(): raise FileNotFoundError(f"Config file {conffile.as_posix()} was not found")
    logger.debug("Found configuration file")
    with conffile.open('r') as fp:
        logger.debug("Reading configuration file")

        if conf_format == 'json':
            logger.debug('Using json')
            conf = json.load(fp)["LMPR"]
        else:
            logger.debug('Using toml')
            conf = toml.load(fp)["LMPR"]
    return conf


def main() -> int:
    logger = minilog("LMPRun")

    parser = argparse.ArgumentParser("LMPRun")
    parser.add_argument("module", action="store", type=str)
    parser.add_argument('--cwd', action='store', type=str, default=None)
    parser.add_argument('-f', '--format', choices=['toml', 'json'], type=str, default='toml')
    parser.add_argument('-c', '--conf', action='store', type=str, default=None)
    parser.add_argument('-r', '--rid', action='store', type=int, default=None)
    parser.add_argument('-t', '--ptag', action='store', type=int, default=None)
    args = parser.parse_args()

    if args.cwd is not None: os.chdir(Path(args.cwd).resolve())

    cwd = Path.cwd()
    slf = cwd / "slurm"
    conffile = Path(args.conf).resolve() if args.conf is not None else cwd / "conf.toml"
    ptag: int = args.ptag if args.ptag is not None else round(time.time())
    rid: int = args.rid if args.rid is not None else 0
    if slf.exists():
        if (slf / ".rid").exists():
            with (slf / ".rid").open('r+') as fp:
                rid = int(fp.read()) + 1 if args.rid is None else rid
                fp.write(str(rid))
        else:
            with (slf / ".rid").open('w') as fp: fp.write(str(rid))
    else:
        slf.mkdir()
        with (slf / ".rid").open('w') as fp: fp.write(str(rid))

    conf = loadconf(conffile, args.format, logger.getChild("loadconf"))
    conf[pysbatch_ng.cs.fields.folder] = slf.as_posix()
    conf[pysbatch_ng.cs.fields.executable] = "LMPResume"
    conf[pysbatch_ng.cs.fields.args] = f"{Path(args.module).resolve().as_posix()} --cwd={cwd.as_posix()}"
    jobid = pysbatch_ng.sbatch.run(cwd, logger.getChild("submitter"), confdict(conf), rid)

    spoll_conf: Dict[str, Union[str, int, bool, Path]] = {}
    spoll_conf[pysbatch_ng.cs.fields.debug] = True
    spoll_conf[pysbatch_ng.cs.fields.cwd] = cwd.as_posix()
    spoll_conf[pysbatch_ng.cs.fields.jobid] = jobid
    spoll_conf[pysbatch_ng.cs.fields.ptag] = ptag
    spoll_conf[pysbatch_ng.cs.fields.logfolder] = cwd / "slurm"
    spoll_conf[pysbatch_ng.cs.fields.logto] = 'file'
    cmd = f"LMPRun {Path(args.module).resolve().as_posix()} --cwd={cwd.as_posix()} --conf={conffile.as_posix()} --rid={rid+1} --ptag={ptag}"
    if args.format is not None: cmd += f" --format={args.format}"
    spoll_conf[pysbatch_ng.cs.fields.cmd] = cmd
    spoll_conf[pysbatch_ng.cs.fields.every] = 5
    spoll_conf[pysbatch_ng.cs.fields.times_criteria] = 288

    conff = {
        pysbatch_ng.cs.fields.spoll: spoll_conf,
        pysbatch_ng.cs.fields.sbatch: conf
        }

    pysbatch_ng.spoll.run_conf(conff, slf, logger.getChild("spoll"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
