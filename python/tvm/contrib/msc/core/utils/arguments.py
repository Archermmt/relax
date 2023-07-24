# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.core.utils.arguments"""

import os
import json
from typing import List


def dump_list(list_obj: List[str]) -> List[dict]:
    """Dump the string list to dict list.

    Parameters
    ----------
    list_obj: list<string>
        The list with string.

    Returns
    -------
    list_dict: list<dict>
        The list with dict.
    """

    list_dict = []
    for o in list_obj:
        if not isinstance(o, str):
            list_dict.append(o)
            continue
        try:
            value = json.loads(o)
        except json.decoder.JSONDecodeError:
            value = o
        if isinstance(value, dict):
            value = {k: dump_list(v) if isinstance(v, list) else v for k, v in value.items()}
        list_dict.append(value)
    return list_dict


def load_dict(str_dict: str, dump_list=False) -> dict:
    """Load the string/file to dict.

    Parameters
    ----------
    str_dict: string
        The file_path or string object.
    dump_list: bool
        Whether to dump the list into dict

    Returns
    -------
    dict_obj: dict
        The loaded dict.
    """

    if isinstance(str_dict, str) and os.path.isfile(str_dict):
        with open(str_dict, "r") as f:
            dict_obj = json.load(f)
    elif isinstance(str_dict, str):
        dict_obj = json.loads(str_dict)
    if dump_list:
        dict_obj = {k: dump_list(v) if isinstance(v, list) else v for k, v in dict_obj.items()}
    return dict_obj


def dump_dict(dict_obj: dict) -> str:
    """Dump the config to string.

    Parameters
    ----------
    src_dict: dict
        The source dict.

    Returns
    -------
    str_dict: string
        The dumped string.
    """

    if not dict_obj:
        return ""
    return json.dumps({k: int(v) if isinstance(v, bool) else v for k, v in dict_obj.items()})
