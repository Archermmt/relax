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
 * \file src/contrib/msc/framework/tvm/relax_codegen.cc
 */
#include "relax_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void RelaxGraphCodeGen::CodeGenHeader() {
  PyGraphCodeGen<RelaxCodeDenConfig>::CodeGenHeader();
  stack_.line("from tvm import relax");
}

void RelaxGraphCodeGen::CodeGenGraph() {
  stack_.func_def(graph()->name, "tvm.IRModule");
  Array<String> idx_inputs;
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    const auto& idx_input = IdxOutput(pair.first, pair.second);
    stack_.func_arg(idx_input, "relax.Var");
    idx_inputs.push_back(idx_input);
  }
  stack_.func_start().assign_list("inputs", idx_inputs);
  // define weights
  stack_.comment("Define the weights and constant");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    for (const auto& pair : node->weights) {
      const auto& idx_weight = IdxWeight(node, pair.first);
      stack_.call_start("relax.Var")
          .call_str_arg(pair.second->name)
          .call_inplace_start("relax.TensorStructInfo")
          .call_list_arg(pair.second->shape)
          .call_str_arg(pair.second->DtypeName())
          .call_inplace_end()
          .call_end(idx_weight)
          .call_start("inputs.append")
          .call_arg(idx_weight)
          .call_end();
    }
    if (node->optype == "constant") {
      CodeGenNode(node);
      stack_.call_start("inputs.append").call_arg(IdxNode(node)).call_end();
    }
  }
  stack_.comment("Define the module");
  stack_.assign("block_builder", "relax.BlockBuilder()")
      .scope_start("block_builder.function(name=\"" + graph()->name + "\", params=inputs.copy())");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input" || node->optype == "constant") {
      continue;
    }
    int scope_level = CompareScope(node);
    if (scope_level == 1) {
      stack_.scope_start("block_builder.dataflow()");
    } else if (scope_level == -1) {
      stack_.scope_end();
    }
    CodeGenNode(node);
  }
  // end left scopes
  for (size_t i = 0; i < scopes().size() - 1; i++) {
    stack_.scope_end();
  }
  // mark outputs
  stack_.comment("Emit the outputs");
  Array<String> idx_outputs;
  for (const auto& o : graph()->output_names) {
    const auto& pair = graph()->FindProducerAndIdx(o);
    const auto& idx_output = IdxOutput(pair.first, pair.second);
    stack_.call_start("block_builder.emit_output").call_arg(idx_output).call_end(idx_output);
    idx_outputs.push_back(idx_output);
  }
  stack_.scope_end().call_start("block_builder.emit_func_output");
  if (idx_outputs.size() == 1) {
    stack_.call_arg(idx_outputs[0]);
  } else {
    stack_.call_list_arg(idx_outputs);
  }
  stack_.call_end().scope_end().assign("mod", "block_builder.get()").func_end("mod");
}

void RelaxGraphCodeGen::CodeGenInference() {
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    stack_.call_start("relax.Var")
        .call_str_arg(i->alias)
        .call_inplace_start("relax.TensorStructInfo")
        .call_list_arg(i->shape)
        .call_str_arg(i->DtypeName())
        .call_inplace_end()
        .call_end(IdxNode(pair.first));
  }
  stack_.comment("Build Module").call_start(graph()->name);
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    stack_.call_arg(IdxNode(pair.first));
  }
  stack_.call_end("mod");
  String target, device;
  if (config()->test_device == "cpu") {
    target = "llvm";
    device = "tvm.cpu()";
  } else if (config()->test_device == "gpu") {
    target = "cuda";
    device = "tvm.cuda()";
  }
  stack_.comment("Load weights")
      .scope_start("open(\"" + graph()->name + "_params.bin\", \"rb\")", "f")
      .call_start("tvm.runtime.load_param_dict")
      .call_arg("f.read()")
      .call_end("params")
      .scope_end()
      .call_start("tvm.relax.transform.BindParams")
      .call_str_arg("main")
      .call_arg("params")
      .call_end("bind_params")
      .call_start("bind_params")
      .call_arg("mod")
      .call_end("mod")
      .call_start("tvm.target.Target")
      .call_str_arg(target)
      .call_end("target")
      .call_start("relax.build")
      .call_arg("mod")
      .call_arg("target")
      .call_end("ex")
      .call_start("relax.VirtualMachine")
      .call_arg("ex")
      .call_arg(device)
      .call_end("vm")
      .call_start("vm[\"main\"]");
  for (const auto& i : graph()->GetInputs()) {
    stack_.call_arg("inputs[\"" + i->alias + "\"]");
  }
  stack_.call_end("outputs");
}

const Array<Doc> RelaxGraphCodeGen::GetOpCodes(const MSCJoint& node) {
  auto it = GetRelaxOpCodeGens()->find(node->optype);
  ICHECK(it != GetRelaxOpCodeGens()->end())
      << "Unsupported relax op(" << node->optype << "): " << node;
  it->second->Config(node, config());
  return it->second->GetDocs();
}

TVM_REGISTER_GLOBAL("msc.tvm.GetRelaxSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String print_config) -> Map<String, String> {
      RelaxGraphCodeGen codegen = RelaxGraphCodeGen(graph, codegen_config);
      codegen.CodeGen();
      return codegen.GetSources(print_config);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
