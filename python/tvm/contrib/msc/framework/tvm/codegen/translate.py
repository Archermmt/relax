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
"""tvm.contrib.msc.framework.tvm.codegen.translate"""

from typing import Dict, Optional

import tvm
from tvm.relax.transform import BindParams
from tvm.contrib.msc.core.ir import MSCGraph
from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.framework.tvm import _ffi_api


def to_relax(
    graph: MSCGraph,
    weights: Optional[Dict[str, tvm.nd.array]] = None,
    codegen_config: Optional[Dict[str, str]] = None,
    print_config: Optional[Dict[str, str]] = None,
    build_folder: msc_utils.MSCDirectory = None,
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

    sources = _ffi_api.GetRelaxSources(
        graph, msc_utils.dump_dict(codegen_config), msc_utils.dump_dict(print_config)
    )
    build_folder = build_folder or msc_utils.msc_dir(keep_history=False, cleanup=True)
    with build_folder as d:
        for name, source in sources.items():
            d.add_file(name, source)
        inputs = [
            tvm.relax.Var(i.alias, tvm.relax.TensorStructInfo(i.get_shape(), i.dtype_name))
            for i in graph.get_inputs()
        ]
        builder = msc_utils.load_callable(graph.name + ".py:" + graph.name)
        mod = builder(*inputs)
        # load weights
        if weights:
            mod = BindParams("main", weights)(mod)
            with open(d.relpath(graph.name + "_params.bin"), "wb") as f_params:
                f_params.write(tvm.runtime.save_param_dict(weights))
    return mod
