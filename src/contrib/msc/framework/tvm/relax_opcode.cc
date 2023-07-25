/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/contrib/msc/framework/tvm/relax_opcode.cc
 */
#include "relax_opcode.h"

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> RelaxOpCodeGen::GetDocs() {
  RelaxOpCodeStack stack = RelaxOpCodeStack(this);
  CodeGenBuild(stack);
  if (node()->optype != "input" && node()->optype != "constant") {
    stack.call_start("block_builder.emit").call_arg(IdxNode(true));
    if (config()->explicit_name) {
      stack.call_str_arg(node()->name, "name_hint");
    }
    stack.call_end(IdxNode(true));
  }
  return stack.GetDocs();
}

void RelaxOpCodeGen::CodeGenBuild(RelaxOpCodeStack& stack) {
  stack.op_start().op_inputs_arg(false).call_end().op_end();
}

#define RELAX_OP_CODEGEN_METHODS(TypeName) \
  TypeName(const String& func_name) : RelaxOpCodeGen(func_name) {}

class RelaxConstantCodeGen : public RelaxOpCodeGen {
 public:
  RELAX_OP_CODEGEN_METHODS(RelaxConstantCodeGen)
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .call_str_arg(node()->name)
        .call_inplace_start("relax.TensorStructInfo")
        .call_list_arg(node()->OutputAt(0)->shape)
        .call_str_arg(node()->OutputAt(0)->DtypeName())
        .call_inplace_end()
        .call_end()
        .op_end();
  }
};

class RelaxConv1dCodeGen : public RelaxOpCodeGen {
 public:
  RELAX_OP_CODEGEN_METHODS(RelaxConv1dCodeGen)
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_weight_arg("weight")
        .op_list_arg<int>("padding")
        .op_list_arg<int>("strides")
        .op_list_arg<int>("strides")
        .op_arg<int>("groups")
        .op_str_arg("data_layout")
        .op_str_arg("kernel_layout")
        .op_str_arg("out_layout")
        .op_str_arg("out_dtype")
        .op_end();
  }
};

class RelaxReshapeCodeGen : public RelaxOpCodeGen {
 public:
  RELAX_OP_CODEGEN_METHODS(RelaxReshapeCodeGen)
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_list_arg<int>("shape").op_end();
  }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<RelaxOpCodeGen>>>
GetRelaxOpCodeGens() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<RelaxOpCodeGen>>>();
  if (!map->empty()) return map;
  // typed ops
  map->emplace("relax.add", std::make_shared<RelaxOpCodeGen>("relax.op.add"));
  map->emplace("relax.divide", std::make_shared<RelaxOpCodeGen>("relax.op.divide"));
  map->emplace("relax.multiply", std::make_shared<RelaxOpCodeGen>("relax.op.multiply"));
  map->emplace("relax.subtract", std::make_shared<RelaxOpCodeGen>("relax.op.subtract"));

  // common ops
  map->emplace("constant", std::make_shared<RelaxConstantCodeGen>("relax.Var"));
  map->emplace("relax.reshape", std::make_shared<RelaxReshapeCodeGen>("relax.op.reshape"));

  // nn ops
  map->emplace("relax.nn.conv1d", std::make_shared<RelaxConv1dCodeGen>("relax.op.nn.conv1d"));
  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
