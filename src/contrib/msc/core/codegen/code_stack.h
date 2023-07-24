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
 * \file src/contrib/msc/core/codegen/code_stack.h
 * \brief CodeStack for doc printer.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_

#include <tvm/script/printer/doc.h>

#include "../printer/print_utils.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief Inner class for doc stack
 */
class BaseStack {
 public:
  /*!
   * \brief The constructor of CodeStack
   */
  explicit BaseStack() {
    while (!blocks_.empty()) {
      blocks_.pop();
    }
    StartBlock();
  }

  /*! \brief Get the docs*/
  const Array<Doc> GetDocs() const;

 protected:
  /*! \brief Push Id Doc*/
  void Line(const Doc& doc);
  void Line(const String& line = "");

  /*! \brief Push Comment Doc*/
  void Comment(const String& comment = "");

  /*! \brief Push assign Doc*/
  void Assign(const String& lhs, const String& rhs, const String& annotation = "");

  /*! \brief Cache function Doc*/
  void FuncDef(const String& func_name, const String& ret_type = "");

  /*! \brief Cache func argument*/
  void FuncArg(const String& arg, const String& annotation = "", const String& value = "");

  /*! \brief Cache func decorator*/
  void FuncDecorator(const String& decorator);

  /*! \brief Start function body block*/
  void FuncBodyStart();

  /*! \brief End function body block*/
  void FuncBodyEnd(const String& ret_val = "");

  /*! \brief Cache call Doc*/
  void CallStart(const String& call_name);

  /*! \brief Push call or/and assign Doc*/
  void CallEnd(const String& assign = "");

  /*! \brief Cache inplace call Doc*/
  void InplaceStart(const String& call_name);

  /*! \brief Push inplace call or/and assign Doc*/
  void InplaceEnd();

  /*! \brief Cache call argument*/
  void CallArgument(const ExprDoc& value, const String& key = "");

  template <typename T>
  inline void CallArg(T value, const String& key = "") {
    CallArgument(DocUtils::ToDoc(value), key);
  }

  void CallStrArg(const String& value, const String& key = "");

  /*! \brief Cache call list argument*/
  void CallListArg(const Array<ExprDoc>& values, const String& key = "");

  template <typename T>
  inline void CallListArg(const std::vector<T>& values, const String& key = "") {
    CallArgument(DocUtils::ToListDoc(values), key);
  }

  template <typename T>
  inline void CallListArg(const Array<T>& values, const String& key = "") {
    CallArgument(DocUtils::ToListDoc(values), key);
  }

  /*! \brief Push if to cache and start if block*/
  void ConditionIf(const String& predicate);

  /*! \brief Push then branch to cached and start block*/
  void ConditionElse();

  /*! \brief Push else branch to cached*/
  void ConditionEnd();

  /*! \brief Start a new block*/
  void StartBlock();

  /*! \brief End a block*/
  void EndBlock(bool block_docs = true);

 private:
  /*! \brief Check if block left*/
  bool HasBlock() const;

  /*! \brief Get the last the block*/
  const Array<Doc> TopBlock() const;

  /*! \brief Pop last the block*/
  const Array<Doc> PopBlock();

  /*! \brief Check if doc left*/
  bool HasDoc();

  /*! \brief Get the last doc*/
  const Doc TopDoc();

  /*! \brief Pop last doc*/
  const Doc PopDoc();

  /*! \brief Pop last doc with type checked*/
  template <typename TDoc, typename TDocNode>
  const TDoc PopCheckedDoc();

  /*! \brief Push doc*/
  void PushDoc(const Doc& doc);

  /*! \brief The blocks, each has docs array*/
  std::stack<Array<Doc>> blocks_;
};

#define COMMON_WRAPPERS(Stack)                                                                  \
  Stack& line(const Doc& doc) {                                                                 \
    Line(doc);                                                                                  \
    return *this;                                                                               \
  }                                                                                             \
  Stack& line(const String& line = "") {                                                        \
    Line(line);                                                                                 \
    return *this;                                                                               \
  }                                                                                             \
  Stack& comment(const String& comment) {                                                       \
    Comment(comment);                                                                           \
    return *this;                                                                               \
  }                                                                                             \
  Stack& assign(const String& lhs, const String& rhs, const String& annotation = "") {          \
    Assign(lhs, rhs, annotation);                                                               \
    return *this;                                                                               \
  }                                                                                             \
  Stack& start_block() {                                                                        \
    StartBlock();                                                                               \
    return *this;                                                                               \
  }                                                                                             \
  Stack& end_block(bool block_docs = true) {                                                    \
    EndBlock(block_docs);                                                                       \
    return *this;                                                                               \
  }                                                                                             \
  Stack& func_def(const String& func_name, const String& ret_type = "") {                       \
    FuncDef(func_name, ret_type);                                                               \
    return *this;                                                                               \
  }                                                                                             \
  Stack& func_arg(const String& arg, const String& annotation = "", const String& value = "") { \
    FuncArg(arg, annotation, value);                                                            \
    return *this;                                                                               \
  }                                                                                             \
  Stack& func_decorator(const String& decorator) {                                              \
    FuncDecorator(decorator);                                                                   \
    return *this;                                                                               \
  }                                                                                             \
  Stack& func_body_start() {                                                                    \
    FuncBodyStart();                                                                            \
    return *this;                                                                               \
  }                                                                                             \
  Stack& func_body_end(const String& ret_val = "") {                                            \
    FuncBodyEnd(ret_val);                                                                       \
    return *this;                                                                               \
  }                                                                                             \
  Stack& call_start(const String& callee) {                                                     \
    CallStart(callee);                                                                          \
    return *this;                                                                               \
  }                                                                                             \
  Stack& call_end(const String& assign = "") {                                                  \
    CallEnd(assign);                                                                            \
    return *this;                                                                               \
  }                                                                                             \
  Stack& inplace_start(const String& callee) {                                                  \
    InplaceStart(callee);                                                                       \
    return *this;                                                                               \
  }                                                                                             \
  Stack& inplace_end() {                                                                        \
    InplaceEnd();                                                                               \
    return *this;                                                                               \
  }                                                                                             \
  template <typename T>                                                                         \
  Stack& call_arg(T value, const String& key = "") {                                            \
    CallArg(value, key);                                                                        \
    return *this;                                                                               \
  }                                                                                             \
  Stack& call_str_arg(const String& value, const String& key = "") {                            \
    CallStrArg(value, key);                                                                     \
    return *this;                                                                               \
  }                                                                                             \
  Stack& call_list_arg(const Array<ExprDoc>& values, const String& key = "") {                  \
    CallListArg(values, key);                                                                   \
    return *this;                                                                               \
  }                                                                                             \
  template <typename T>                                                                         \
  Stack& call_list_arg(const std::vector<T>& values, const String& key = "") {                  \
    CallListArg(values, key);                                                                   \
    return *this;                                                                               \
  }                                                                                             \
  template <typename T>                                                                         \
  Stack& call_list_arg(const Array<T>& values, const String& key = "") {                        \
    CallListArg(values, key);                                                                   \
    return *this;                                                                               \
  }                                                                                             \
  Stack& cond_if(const String& predicate) {                                                     \
    ConditionIf(predicate);                                                                     \
    return *this;                                                                               \
  }                                                                                             \
  Stack& cond_else() {                                                                          \
    ConditionElse();                                                                            \
    return *this;                                                                               \
  }                                                                                             \
  Stack& cond_end() {                                                                           \
    ConditionEnd();                                                                             \
    return *this;                                                                               \
  }

/*!
 * \brief Stack Doc for common codegen
 */
class CodeStack : public BaseStack {
 public:
  /*!
   * \brief The constructor of CodeStack
   */
  explicit CodeStack() : BaseStack() {}

  COMMON_WRAPPERS(CodeStack)
};

/*!
 * \brief Stack Doc for codes
 */
template <typename CodeGenType>
class OpCodeStack : public BaseStack {
 public:
  /*!
   * \brief The constructor of OpCodeStack
   * \param codegen the OpCodeGen pointer.
   */
  explicit OpCodeStack(const CodeGenType* codegen) : BaseStack() { codegen_ = codegen; }

  COMMON_WRAPPERS(OpCodeStack<CodeGenType>)

  /*! \brief Cache func_call and assign Doc*/
  OpCodeStack<CodeGenType>& op_start(const String& func_name = "auto",
                                     const String& func_ret = "auto") {
    const String& call_name = func_name == "auto" ? codegen_->func_name() : func_name;
    const IdDoc& ret = func_ret == "auto" ? codegen_->IdxNode(true) : IdDoc(func_ret);
    return call_start(call_name, ret);
  }

  /*! \brief Cache attribute as argument*/
  template <typename T>
  OpCodeStack<CodeGenType>& call_attr_arg(const String& attr_key = "", const String& key = "") {
    T attr_val;
    if (codegen_->node()->GetAttr(attr_key, attr_val)) {
      return call_arg(attr_val, key.size() == 0 ? attr_key : key);
    }
    return *this;
  }

  /*! \brief Cache input as argument*/
  OpCodeStack<CodeGenType>& call_input_arg(int idx = 0, const String& key = "") {
    return call_arg(codegen_->IdxInput(idx, false), key);
  }

  /*! \brief Cache inputs as argument*/
  OpCodeStack<CodeGenType>& call_inputs_arg(const String& key = "", bool as_list = true) {
    Array<ExprDoc> inputs;
    for (size_t i = 0; i < codegen_->node()->inputs.size(); i++) {
      inputs.push_back(codegen_->IdxInput(i, false));
    }
    if (as_list) {
      return call_list_arg(inputs, key);
    }
    for (const auto& i : inputs) {
      call_arg(i, key);
    }
    return *this;
  }

  /*! \brief Cache output as argument*/
  OpCodeStack<CodeGenType>& call_output_arg(int idx = 0, const String& key = "") {
    return call_arg(codegen_->IdxOutput(idx, false), key);
  }

  /*! \brief Cache weight as argument*/
  OpCodeStack<CodeGenType>& call_weight_arg(const String& wtype, const String& key = "") {
    return call_arg(codegen_->IdxWeight(wtype, false), key);
  }

 private:
  CodeGenType* codegen_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CODE_STACK_H_
