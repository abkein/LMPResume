#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 20-06-2024 19:16:58

from pathlib import Path
from typing import Dict, List

import pandas
import lammps.formats


def extract_thermo(file: Path) -> None:
    outfile = Path("thermo.csv")
    thermo = lammps.formats.LogFile(file.as_posix())
    runs: List[Dict[str, List[float]]] = thermo.runs
    keyhash = 0
    header: List[str] = []
    thermolist: List[List[float]] = []
    for el in runs:
        if hash(tuple(el.keys())) != keyhash:
            if len(thermolist) > 0:
                pandas.DataFrame(thermolist).to_csv(outfile.as_posix(), mode='a+', encoding='utf-8', index=False, header=header)
            keyhash = hash(tuple(el.keys()))
            header = list(el.keys())
        lst: List[float] = []
        for val in el.values():
            lst.append(val[0])
        thermolist.append(lst)
    if len(thermolist) > 0:
        pandas.DataFrame(thermolist).to_csv(outfile.as_posix(), mode='a+', encoding='utf-8', index=False, header=header)


if __name__ == "__main__":
    pass

