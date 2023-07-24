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
  stack_.line("from tvm impot relax");
}

void RelaxGraphCodeGen::CodeGenGraph() {
  stack_.func_def(graph()->name, "relax.Function");
  for (const auto& i : graph()->GetInputs()) {
    stack_.func_arg(i->alias, "relax.Var");
  }
  stack_.func_body_start();
  stack_.func_body_end("outputs");
}

void RelaxGraphCodeGen::CodeGenInference() {}

const Array<Doc> RelaxGraphCodeGen::GetOpCodes(const MSCJoint& node,
                                               const std::shared_ptr<RelaxCodeDenConfig>& config) {
  Array<Doc> docs;
  return docs;
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
