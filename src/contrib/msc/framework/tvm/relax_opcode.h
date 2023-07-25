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
 * \file src/contrib/msc/framework/tvm/relax_opcode.h
 * \brief Relax codegen for MSCJoint.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TVM_RELAX_OPCODE_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TVM_RELAX_OPCODE_H_

#include "../../core/codegen/base_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen config for relax
 */
struct RelaxCodeDenConfig {
  bool explicit_name{true};
  CODEGEN_CONFIG_MEMBERS
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "explicit_name") {
        reader->Read(&explicit_name);
      } else {
        CODEGEN_CONFIG_PARSE
      }
    }
  }
};

/*!
 * \brief CodeStack for relax op
 */
class RelaxOpCodeGen;
class RelaxOpCodeStack : public OpCodeStack<RelaxOpCodeGen> {
 public:
  /*!
   * \brief The constructor of RelaxOpCodeStack
   * \param codegen the OpCodeGen pointer.
   */
  explicit RelaxOpCodeStack(RelaxOpCodeGen* codegen) : OpCodeStack<RelaxOpCodeGen>(codegen) {}
};

/*!
 * \brief CodeGen for relax op
 */
class RelaxOpCodeGen : public BaseOpCodeGen<RelaxCodeDenConfig> {
 public:
  /*!
   * \brief The constructor of BaseOpDocsifier
   * \param func_name the function name for the node.
   * \param config the config json for the node.
   */
  explicit RelaxOpCodeGen(const String& func_name) : BaseOpCodeGen<RelaxCodeDenConfig>(func_name) {}

  /*! \brief Convert node to docs*/
  const Array<Doc> GetDocs() final;

  /*! \brief Convert op build*/
  virtual void CodeGenBuild(RelaxOpCodeStack& stack);
};

/*!
 * \brief Get the map of available RelaxOpCodeGen, use optype as key
 * \return Map of <string, RelaxOpCodeGen>
 */
const std::shared_ptr<std::unordered_map<String, std::shared_ptr<RelaxOpCodeGen>>>
GetRelaxOpCodeGens();

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TVM_RELAX_OPCODE_H_