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

import tvm
from tvm.relax.transform import BindParams
from tvm.contrib.msc.core.transform import transform
from tvm.contrib.msc.core import utils as msc_utils


def from_relax(mod, params=None, trans_config=None, build_config=None):
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
    print(msc_utils.show_function(mod))
    info = msc_utils.get_span_attrs(mod)
    print("[TMINFO] info " + str(info))
    return mod
