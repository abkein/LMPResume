#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# Copyright (c) 2023-2024 Perevoshchikov Egor
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Last modified: 16-06-2024 12:30:40

import json
from pathlib import Path
from typing import Dict, Any, Type, List

from .meta import SerialProtocol


def MakeDecoder(types2dec: List[Type]):
    class UtilJSONDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs) -> None:
            json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

        def object_hook(self, obj: Dict[str, Any]):
            if "__type" in obj:
                _type: Type[SerialProtocol]
                for _type in types2dec:
                    if obj["__type"] == _type.__name__:  # type: ignore
                        return _type.__rejs__(_type, **obj)  # type: ignore

            return obj
    return UtilJSONDecoder

class UniversalJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SerialProtocol):
            return obj.__json__()
        elif isinstance(obj, Path):
            return obj.as_posix()
        elif hasattr(obj, "__json__") and hasattr(type(obj), "__rejs__"):
            return obj.__json__()
        else:
            return super().default(obj)
