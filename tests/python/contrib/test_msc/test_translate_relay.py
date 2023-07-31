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
import numpy as np
import torch
from torch import fx
from torch.nn import Module

import tvm.testing
from tvm.relax.frontend.torch import from_fx
from tvm.relay.frontend import from_pytorch
from tvm.contrib.msc.core.ir import translate
from tvm.contrib.msc.core import utils as msc_utils


def verify_model(torch_model, input_info):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        expected = from_fx(graph_model, input_info)
    # graph from relay
    input_datas = [np.ones(i[0].astype(i[1])) for i in input_info]
    t_inputs = [torch.from_numpy(i) for i in input_datas]
    scripted_model = torch.jit.trace(torch_model, *t_inputs).eval()  # type: ignore
    shape_list = [("input" + str(idx), i) for idx, i in enumerate(input_info)]
    relay_mod, params = from_pytorch(scripted_model, shape_list)
    graph, weights = translate.from_relay(relay_mod, params)
    print("relay graph " + str(graph))
    # to relax
    build_folder = msc_utils.msc_dir("msc_test", cleanup=False)
    mod = translate.to_relax(
        graph, weights, codegen_config={"explicit_name": False}, build_folder=build_folder
    )
    print("Compare {}\n with {}".format(expected, mod))
    tvm.ir.assert_structural_equal(mod, expected)


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
    # verify_model(Conv1D2(), input_info)


def test_conv2d():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Conv2D1(), input_info)
    verify_model(Conv2D2(), input_info)


def test_linear():
    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            return self.linear(input)

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, input):
            return self.linear(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(Dense1(), input_info)
    verify_model(Dense2(), input_info)


def test_bmm():
    class BMM(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.bmm(x, y)

    input_info = [((4, 128, 256), "float32"), ((4, 256, 512), "float32")]
    verify_model(BMM(), input_info)


def test_relu():
    class ReLU(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input):
            return self.relu(input)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU(), input_info)


def test_relu6():
    class ReLU6(Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, input):
            return self.relu6(input)

    input_info = [([10, 10], "float32")]
    verify_model(ReLU6(), input_info)


def test_maxpool2d():
    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, input):
            return self.pool(input)

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(MaxPool2d(), input_info)
    verify_model(MaxPool2d2(), input_info)
    verify_model(MaxPool2d3(), input_info)


def test_avgpool2d():
    class AvgPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class AvgPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True)

        def forward(self, input):
            return self.pool(input)

    class AvgPool2d3(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(
                input, kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True
            )

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AvgPool2d(), input_info)
    # verify_model(AvgPool2d2(), input_info)
    # verify_model(AvgPool2d3(), input_info)


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2d0(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, input):
            return self.pool(input)

    class AdaptiveAvgPool2d1(Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool2d(input, [10, 10])

    input_info = [([1, 3, 10, 10], "float32")]
    verify_model(AdaptiveAvgPool2d0(), input_info)
    verify_model(AdaptiveAvgPool2d1(), input_info)


if __name__ == "__main__":
    # tvm.testing.main()
    test_conv1d()
