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
import torch
from torch import fx
from torch.nn import Module

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.contrib.msc.core.ir import translate
from tvm.contrib.msc.core import utils as msc_utils


def _translate(expected):
    graph, weights = translate.from_relax(expected)
    build_folder = msc_utils.msc_dir("msc_test", cleanup=True)
    return translate.to_relax(
        graph, weights, codegen_config={"explicit_name": False}, build_folder=build_folder
    )


def verify_model(torch_model, input_info):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        expected = from_fx(graph_model, input_info)
    tvm.ir.assert_structural_equal(_translate(expected), expected)


def test_conv1d():
    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    input_info = [([1, 3, 10], "float32")]
    verify_model(Conv1D1(), input_info)
    verify_model(Conv1D2(), input_info)


if __name__ == "__main__":
    tvm.testing.main()
