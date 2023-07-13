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
 * \file tvm/contrib/msc/core/transform/set_expr_name.cc
 * \brief Pass for setting name for call and constant.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

class ExprNameSetter : public ExprVisitor {
 public:
  void VisitExpr_(const ConstantNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& unique_name = GetUniqueName(GetRef<Constant>(op), "const");
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const TupleNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& unique_name = GetUniqueName(GetRef<Tuple>(op), "tuple");
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& tuple_name = SpanUtils::GetAttr(op->tuple->span, "name");
    const String& unique_name = tuple_name + "." + std::to_string(op->index);
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const std::string& op_name = Downcast<Op>(op->op)->name;
    int rpos = op_name.rfind(".");
    const String& unique_name = GetUniqueName(GetRef<Call>(op), op_name.substr(rpos + 1));
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
    // set constant consumer && master
    const auto& input_types = ExprUtils::GetInputTypes(GetRef<Call>(op));
    for (size_t i = 0; i < input_types.size(); i++) {
      if (input_types[i] == "input") {
        continue;
      }
      if (const auto* c_node = op->args[i].as<ConstantNode>()) {
        const String& const_name = SpanUtils::GetAttr(c_node->span, "name");
        if (constant_consumers_.count(const_name)) {
          op->span = SpanUtils::SetAttr(op->span, "master", constant_consumers_[const_name]);
        } else {
          constant_consumers_.Set(const_name, unique_name);
        }
      }
    }
  }

 private:
  const String GetUniqueName(const Expr& expr, const String& name_hint) {
    String expr_name = SpanUtils::GetAttr(expr->span, "name");
    if (expr_name.size() == 0) {
      expr_name = name_hint;
    }
    if (!setted_names_.count(expr_name)) {
      setted_names_.Set(expr_name, expr);
      return expr_name;
    }
    if (setted_names_[expr_name] == expr) {
      return expr_name;
    }
    int cnt = 1;
    while (setted_names_.count(expr_name + "_" + std::to_string(cnt)) &&
           setted_names_[expr_name + "_" + std::to_string(cnt)] != expr) {
      cnt++;
    }
    expr_name = expr_name + "_" + std::to_string(cnt);
    if (!setted_names_.count(expr_name)) {
      setted_names_.Set(expr_name, expr);
    }
    return expr_name;
  }

  Map<String, Expr> setted_names_;
  Map<String, String> constant_consumers_;
};  // class ExprNameSetter

Expr SetExprName(const Expr& e) {
  ExprNameSetter().VisitExpr(e);
  return e;
}

namespace transform {

Pass SetExprName() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(SetExprName(f)); };
  return CreateFunctionPass(pass_func, 1, "SetExprName", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SetExprName").set_body_typed(SetExprName);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
