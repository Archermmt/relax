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
"""tvm.contrib.msc.core.ir.translate"""

from typing import Dict, Optional, Tuple

import tvm
from tvm.relax.transform import BindParams
from tvm.contrib.msc.core.transform import transform
from tvm.contrib.msc.core import _ffi_api
from tvm.contrib.msc.core import utils as msc_utils
from .graph import MSCGraph


def from_relax(
    mod: tvm.IRModule,
    params: Optional[Dict[str, tvm.nd.array]] = None,
    trans_config: Optional[Dict[str, str]] = None,
    build_config: Optional[Dict[str, str]] = None,
) -> Tuple[MSCGraph, Dict[str, tvm.nd.array]]:
    """Change IRModule to MSCGraph.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    params: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    trans_config: dict
        The config for transfrorm IRModule.
    build_config: dict
        The config for build MSCGraph.

    Returns
    -------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The weights from the IRModule.
    """

    trans_config = trans_config or {}
    build_config = build_config or {}
    if params:
        mod = BindParams("main", params)(mod)
    passes = [
        transform.SetExprName(),
        transform.SetExprLayout(trans_config.get("allow_layout_missing", True)),
    ]
    mod = tvm.transform.Sequential(passes)(mod)
    graph = _ffi_api.BuildFromRelax(mod, "main", msc_utils.dump_dict(build_config))
    weights = _ffi_api.GetRelaxWeights(mod, "main")
    return graph, weights


def to_relax(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: str = None,
):
    """Change MSCGraph to IRModule.

    Parameters
    ----------
    graph: tvm.contrib.msc.core.ir.MSCGraph
        The translated graph.
    weights: dict of <string:tvm.ndarray>
        The parameters of the IRModule.
    codegen_config: dict
        The config for codegen.
    print_config: dict
        The config for print.
    build_folder: str
        The folder for saving scripts and datas.

    Returns
    -------
    mod: IRModule
        The IRModule of relax.
    """

    func = tvm.get_global_func("msc.tvm.GetRelaxSources")
    assert func, "Can not find msc.tvm.GetRelaxSources, please build with MSC"
    sources = func(graph, msc_utils.dump_dict(codegen_config), msc_utils.dump_dict(print_config))
    print("[TMINFO] get sources " + str(sources))
    return graph
