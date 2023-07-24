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
 * \file src/contrib/msc/core/codegen/codegen_utils.h
 * \brief Common utilities for print.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_UTILS_H_

#include <tvm/script/printer/doc.h>

#include "../ir/graph.h"
#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief Utils for CodeGen.
 */
class CodeGenUtils {
 public:
  /*!
   * \brief Get indexed node string.
   * \return The String.
   */
  TVM_DLL static const String IdxNode(const MSCJoint& node, const String& prefix,
                                      const String& suffix = "");

  /*!
   * \brief Get indexed output string.
   * \return The String.
   */
  TVM_DLL static const String IdxOutput(const MSCJoint& node, const String& prefix, int idx = 0,
                                        const String& suffix = "");

  /*!
   * \brief Get indexed input string.
   * \return The String.
   */
  TVM_DLL static const String IdxInput(const MSCJoint& node, const String& prefix, int idx = 0,
                                       const String& suffix = "");

  /*!
   * \brief Get indexed weight string.
   * \return The String.
   */
  TVM_DLL static const String IdxWeight(const MSCJoint& node, const String& wtype,
                                        const String& suffix = "");

  /*!
   * \brief Get comment of a node.
   * \return The String.
   */
  TVM_DLL static const String CommentNode(const MSCJoint& node, const String& prefix);

  /*!
   * \brief Convert the docs to Stmts.
   * \return The Stmts.
   */
  TVM_DLL static const Array<StmtDoc> ToStmts(const Array<Doc>& docs);

  /*!
   * \brief Convert the docs to StmtBlock.
   * \return The StmtBlockDoc.
   */
  TVM_DLL static const StmtBlockDoc ToStmtBlock(const Array<Doc>& docs);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CODEGEN_UTILS_H_
