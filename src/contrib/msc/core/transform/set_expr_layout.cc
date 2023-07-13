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
 * \file tvm/contrib/msc/core/transform/set_var_layout.cc
 * \brief Pass for setting layout for var and constant.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

// #include "../../../../relax/transform/infer_layout_utils.h"
// #include "../../../../relax/transform/utils.h"
#include "../utils.h"
#include "layout_utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

NLayout InferNLayout(const Expr& expr, const VarLayoutMap& var_layout_map) {
  if (expr.as<VarNode>() && var_layout_map.count(Downcast<Var>(expr))) {
    return GetNLayout(var_layout_map, expr);
  }
  return LayoutUtils::GetNLayout(expr);
}

LayoutDecision InferLayoutDecision(const Expr& expr, const VarLayoutMap& var_layout_map) {
  NLayout nlayout = InferNLayout(expr, var_layout_map);
  ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << expr;
  return nlayout.LeafValue();
}

std::tuple<int64_t, int64_t> AccumulateMatch(const std::vector<int64_t>& in_shape,
                                             const std::vector<int64_t>& out_shape, size_t in_start,
                                             size_t out_start) {
  // find input position in_pos and output position out_pos
  // mul(in_shape[in_start:in_ops])==mul(out_shape[out_start:out_pos])
  int64_t in_pos = -1;
  int64_t out_pos = -1;
  int64_t in_accumulate = 1;
  int64_t out_accumulate = 1;
  for (size_t i = in_start; i < in_shape.size(); i++) {
    in_accumulate *= in_shape[i];
    for (size_t j = out_start; j < out_shape.size(); j++) {
      out_accumulate *= out_shape[j];
      if (in_accumulate == out_accumulate) {
        in_pos = i;
        out_pos = j;
        break;
      } else if (out_accumulate > in_accumulate) {
        break;
      }
    }
    if (in_pos >= 0) {
      break;
    }
  }
  // append tailed 1s
  if (in_pos >= 0) {
    while (in_pos < in_shape.size() - 1 && in_shape[in_pos + 1] == 1) {
      in_pos++;
    }
    while (out_pos < out_shape.size() - 1 && out_shape[out_pos + 1] == 1) {
      out_pos++;
    }
  }
  return std::make_tuple(in_pos, out_pos);
}

std::vector<size_t> InferReduceAxes(const Array<PrimExpr>& input_shape,
                                    const Array<PrimExpr>& output_shape) {
  std::vector<size_t> reduce_axes, out_axes;
  std::vector<int64_t> in_shape, out_shape;
  for (const auto& s : input_shape) {
    in_shape.push_back(Downcast<Integer>(s)->value);
  }
  for (const auto& s : output_shape) {
    out_shape.push_back(Downcast<Integer>(s)->value);
  }
  size_t start = 0;
  while (start < in_shape.size() && out_axes.size() < out_shape.size()) {
    if (in_shape[start] == out_shape[out_axes.size()]) {
      out_axes.push_back(start);
      start++;
    } else {
      int64_t in_pos, out_pos;
      size_t out_start = out_axes.size();
      std::tie(in_pos, out_pos) = AccumulateMatch(in_shape, out_shape, start, out_start);
      if (in_pos == -1) {
        return std::vector<size_t>();
      }
      for (size_t i = out_start; i < out_pos; i++) {
        out_axes.push_back(i + 1);
      }
      start = in_pos + 1;
    }
  }
  if (out_axes.size() != out_shape.size()) {
    return std::vector<size_t>();
  }
  std::set<size_t> out_axes_set;
  for (const auto& a : out_axes) {
    out_axes_set.insert(a);
  }
  for (size_t i = 0; i < in_shape.size(); i++) {
    if (!out_axes_set.count(i)) {
      reduce_axes.push_back(i);
    }
  }
  return reduce_axes;
}

std::vector<size_t> InferExpandAxes(const Array<PrimExpr>& input_shape,
                                    const Array<PrimExpr>& output_shape) {
  std::vector<size_t> expand_axes;
  std::vector<int64_t> in_shape, out_shape;
  for (const auto& s : input_shape) {
    in_shape.push_back(Downcast<Integer>(s)->value);
  }
  for (const auto& s : output_shape) {
    out_shape.push_back(Downcast<Integer>(s)->value);
  }
  size_t start = 0;
  while (start < in_shape.size() && expand_axes.size() + in_shape.size() < out_shape.size()) {
    if (in_shape[start] == out_shape[start + expand_axes.size()]) {
      start++;
    } else {
      int64_t in_pos, out_pos;
      size_t out_start = start + expand_axes.size();
      std::tie(in_pos, out_pos) = AccumulateMatch(in_shape, out_shape, start, out_start);
      if (in_pos == -1) {
        return std::vector<size_t>();
      }
      size_t expand_size = out_pos - in_pos - expand_axes.size();
      for (size_t i = out_start; i < expand_size; i++) {
        expand_axes.push_back(i + 1);
      }
      start = in_pos + 1;
    }
  }
  if (expand_axes.size() + in_shape.size() != out_shape.size()) {
    return std::vector<size_t>();
  }
  return expand_axes;
}

// Forward Infer
InferLayoutOutput ForwardInferLayoutCommon(const Call& call,
                                           const Map<String, Array<String>>& desired_layouts,
                                           const VarLayoutMap& var_layout_map) {
  Array<NLayout> input_layouts;
  NLayout layout_hint;
  for (const auto& arg : call->args) {
    const auto& in_layout = InferNLayout(arg, var_layout_map);
    if (in_layout.IsLeaf() && in_layout.LeafValue()->layout.defined()) {
      layout_hint = in_layout;
    }
    input_layouts.push_back(in_layout);
  }

  if (!layout_hint.defined()) {
    return InferLayoutOutput();
  }
  std::vector<NLayout> output_layouts;
  auto sinfo = GetStructInfo(call);
  if (sinfo.as<TensorStructInfoNode>()) {
    output_layouts.push_back(layout_hint);
  } else if (const auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
    for (size_t i = 0; i < tuple_sinfo->fields.size(); i++) {
      output_layouts.push_back(layout_hint);
    }
  } else {
    return InferLayoutOutput();
  }
  return InferLayoutOutput(input_layouts, {output_layouts}, Attrs());
}

InferLayoutOutput InferLayoutConv(const Call& call,
                                  const Map<String, Array<String>>& desired_layouts,
                                  const VarLayoutMap& var_layout_map) {
  LayoutDecision data_layout, kernel_layout, out_layout;
  const String& op_name = Downcast<Op>(call->op)->name;
  if (op_name == "relax.nn.conv1d") {
    const auto* attrs = call->attrs.as<Conv1DAttrs>();
    data_layout = LayoutDecision(attrs->data_layout);
    kernel_layout = LayoutDecision(attrs->kernel_layout);
    out_layout = LayoutDecision(attrs->out_layout);
  } else if (op_name == "relax.nn.conv2d") {
    const auto* attrs = call->attrs.as<Conv2DAttrs>();
    data_layout = LayoutDecision(attrs->data_layout);
    kernel_layout = LayoutDecision(attrs->kernel_layout);
    out_layout = LayoutDecision(attrs->out_layout);
  }
  return InferLayoutOutput({data_layout, kernel_layout}, {out_layout}, Attrs());
}

InferLayoutOutput ForwardInferLayoutReshape(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  LayoutDecision input_layout = InferLayoutDecision(call->args[0], var_layout_map);
  if (!input_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  LayoutDecision output_layout;

  return InferLayoutOutput({input_layout}, {output_layout}, Attrs());
}

TVM_REGISTER_OP("relax.nn.conv1d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", InferLayoutConv);
TVM_REGISTER_OP("relax.nn.conv2d")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", InferLayoutConv);

TVM_REGISTER_OP("relax.reshape")
    .set_attr<FRelaxInferLayout>("FMSCForwardInferLayout", ForwardInferLayoutReshape);

// Backward Infer
InferLayoutOutput BackwardInferLayoutCommon(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  NLayout output_layout = InferNLayout(call, var_layout_map);
  if (!output_layout.LeafValue()->layout.defined()) {
    return InferLayoutOutput();
  }
  Array<NLayout> input_layouts;
  for (const auto& arg : call->args) {
    if (arg.as<VarNode>() && var_layout_map.count(Downcast<Var>(arg))) {
      input_layouts.push_back(GetNLayout(var_layout_map, arg));
    } else {
      input_layouts.push_back(output_layout);
    }
  }
  return InferLayoutOutput(input_layouts, {output_layout}, Attrs());
}

InferLayoutOutput BackwardInferLayoutBinary(const Call& call,
                                            const Map<String, Array<String>>& desired_layouts,
                                            const VarLayoutMap& var_layout_map) {
  return BackwardInferLayoutCommon(call, desired_layouts, var_layout_map);
}

InferLayoutOutput BackwardInferLayoutReshape(const Call& call,
                                             const Map<String, Array<String>>& desired_layouts,
                                             const VarLayoutMap& var_layout_map) {
  LayoutDecision output_layout = InferLayoutDecision(call, var_layout_map);
  if (!output_layout->layout.defined()) {
    return InferLayoutOutput();
  }
  Array<PrimExpr> empty;
  const auto& input_shape =
      Downcast<TensorStructInfo>(GetStructInfo(call->args[0]))->GetShape().value_or(empty);
  const auto& output_shape =
      Downcast<TensorStructInfo>(GetStructInfo(call))->GetShape().value_or(empty);
  if (input_shape.size() == 0 || output_shape.size() == 0) {
    return InferLayoutOutput();
  }
  LayoutDecision input_layout;
  if (input_shape.size() == output_shape.size()) {
    input_layout = output_layout;
  } else if (input_shape.size() > output_shape.size()) {
    const auto& reduce_axes = InferReduceAxes(input_shape, output_shape);
    if (reduce_axes.size() == 0) {
      return InferLayoutOutput();
    }
    input_layout = LayoutUtils::ExpandLayout(output_layout, reduce_axes);
  } else {
    const auto& expand_axes = InferExpandAxes(input_shape, output_shape);
    if (expand_axes.size() == 0) {
      return InferLayoutOutput();
    }
    input_layout = LayoutUtils::ReduceLayout(output_layout, expand_axes);
  }
  return InferLayoutOutput({input_layout, LayoutDecision("O")}, {output_layout}, Attrs());
}

TVM_REGISTER_OP("relax.nn.conv1d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", InferLayoutConv);
TVM_REGISTER_OP("relax.nn.conv2d")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", InferLayoutConv);
TVM_REGISTER_OP("relax.reshape")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutReshape);

TVM_REGISTER_OP("relax.add")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.divide")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.floor_divide")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.multiply")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.power")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.subtract")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.greater")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.greater_equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.less")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.less_equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.not_equal")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.maximum")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.minimum")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_and")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_or")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.logical_xor")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_and")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_or")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);
TVM_REGISTER_OP("relax.bitwise_xor")
    .set_attr<FRelaxInferLayout>("FMSCBackwardInferLayout", BackwardInferLayoutBinary);

class LayoutInfer : public ExprVisitor {
 public:
  explicit LayoutInfer() { Reset(); }

  void Reset() {
    infered_ = false;
    var_map_.clear();
    ordered_exprs_.clear();
  }

  void RecordExpr(const Var& var, const Expr& expr) {
    var_map_.Set(var, expr);
    ordered_exprs_.push_back(expr);
  }

  Expr Infer(const Expr& expr) {
    Reset();
    ForwardInfer(expr);
    BackwardInfer();
    return expr;
  }

  void ForwardInfer(const Expr& expr) { ExprVisitor::VisitExpr(expr); }

  void BackwardInfer() {
    for (size_t e_idx = ordered_exprs_.size(); e_idx > 0; e_idx--) {
      const Expr& expr = ordered_exprs_[e_idx - 1];
      if (const auto* t_node = expr.as<TupleNode>()) {
        continue;
      }
      if (const auto* t_node = expr.as<TupleGetItemNode>()) {
        continue;
      }
      if (!expr.as<CallNode>()) {
        continue;
      }
      const Call& call = Downcast<Call>(expr);
      size_t infered_num = 0;
      for (const auto& arg : call->args) {
        if (arg.as<VarNode>() && var_map_.count(Downcast<Var>(arg))) {
          if (LayoutUtils::LayoutInfered(var_map_[Downcast<Var>(arg)]) > 0) {
            infered_num++;
          }
        } else if (LayoutUtils::LayoutInfered(arg)) {
          infered_num++;
        }
      }
      if (call->args.size() == 0 || infered_num == call->args.size() || !call->op.as<OpNode>() ||
          LayoutUtils::HasUnknownDimTensor(call->args)) {
        continue;
      }
      const OpNode* op_node = call->op.as<OpNode>();
      if (op_node == nullptr) {
        continue;
      }
      // Infer by op_node
      Op op = Downcast<Op>(GetRef<Op>(op_node));
      InferLayoutOutput infered_layout;
      const auto msc_infer_map = Op::GetAttrMap<FRelaxInferLayout>("FMSCBackwardInferLayout");
      if (msc_infer_map.count(op)) {
        FRelaxInferLayout f = msc_infer_map[op];
        infered_layout = f(call, Map<String, Array<String>>(), var_layout_map_);
      }
      if (infered_layout.defined() && infered_layout->input_layouts.size() == call->args.size()) {
        for (size_t i = 0; i < infered_layout->input_layouts.size(); i++) {
          const auto& in_layout = infered_layout->input_layouts[i];
          if (call->args[i].as<VarNode>() && var_map_.count(Downcast<Var>(call->args[i]))) {
            const auto& p_expr = var_map_[Downcast<Var>(call->args[i])];
            if (!LayoutUtils::LayoutInfered(p_expr) && LayoutUtils::SetLayout(p_expr, in_layout)) {
              infered_ = true;
            }
          } else if (LayoutUtils::SetLayout(call->args[i], in_layout)) {
            infered_ = true;
          }
        }
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    bool infer_outputs = true;
    RecordExpr(binding->var, GetRef<Call>(call_node));
    if (LayoutUtils::LayoutInfered(GetRef<Call>(call_node))) {
      infer_outputs = false;
    }
    if (call_node->args.size() == 0 || !call_node->op.as<OpNode>() ||
        LayoutUtils::HasUnknownDimTensor(call_node->args)) {
      infer_outputs = false;
    }
    const OpNode* op_node = call_node->op.as<OpNode>();
    if (op_node == nullptr) {
      infer_outputs = false;
    }
    if (infer_outputs) {
      // infer layouts
      Op op = Downcast<Op>(GetRef<Op>(op_node));
      InferLayoutOutput infered_layout;
      const auto msc_infer_map = Op::GetAttrMap<FRelaxInferLayout>("FMSCForwardInferLayout");
      const auto relax_infer_map = Op::GetAttrMap<FRelaxInferLayout>("FRelaxInferLayout");
      if (msc_infer_map.count(op)) {
        FRelaxInferLayout f = msc_infer_map[op];
        infered_layout = f(GetRef<Call>(call_node), Map<String, Array<String>>(), var_layout_map_);
      } else if (relax_infer_map.count(op)) {
        FRelaxInferLayout f = relax_infer_map[op];
        infered_layout = f(GetRef<Call>(call_node), Map<String, Array<String>>(), var_layout_map_);
      }
      if (infered_layout.defined() && infered_layout->output_layouts.size() == 1) {
        var_layout_map_[binding->var] = infered_layout->output_layouts[0];
        if (LayoutUtils::SetLayout(GetRef<Call>(call_node), infered_layout->output_layouts[0])) {
          infered_ = true;
        }
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    std::vector<NLayout> input_layout;
    for (const auto& field : val->fields) {
      if (binding->var->IsInstance<DataflowVarNode>()) {
        // Df var: Use the current realized layout to group the tuple;
        input_layout.push_back(GetNLayout(var_layout_map_, field));
      } else {
        // Global var: Use the initial layout to group the tuple;
        input_layout.push_back(InitialNLayout(field));
      }
    }
    if (IsNestedTensor(binding->var)) {
      var_layout_map_[binding->var] = input_layout;
    }
    RecordExpr(binding->var, GetRef<Tuple>(val));
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    NLayout input_layout = binding->var->IsInstance<DataflowVarNode>()
                               ? GetNLayout(var_layout_map_, val->tuple)
                               : InitialNLayout(val->tuple);
    var_layout_map_[binding->var] = input_layout.NestedArray()[val->index];
    RecordExpr(binding->var, GetRef<TupleGetItem>(val));
  }

  bool infered() { return infered_; }

 private:
  bool infered_;
  Map<Var, Expr> var_map_;
  Array<Expr> ordered_exprs_;
  std::unordered_map<Var, NLayout, ObjectPtrHash, ObjectPtrEqual> var_layout_map_;
};  // class LayoutInfer

class LayoutChecker : public ExprVisitor {
 public:
  LayoutChecker() { missing_num_ = 0; }

  void Check(const Expr& expr) {
    ExprVisitor::VisitExpr(expr);
    ICHECK_EQ(missing_num_, 0) << "Some layout is missing";
  }

  void VisitExpr_(const CallNode* call) final {
    ExprVisitor::VisitExpr_(call);
    if (!LayoutUtils::LayoutInfered(GetRef<Call>(call))) {
      missing_num_++;
    }
  }

  void VisitExpr_(const ConstantNode* cn) final {
    ExprVisitor::VisitExpr_(cn);
    if (!LayoutUtils::LayoutInfered(GetRef<Constant>(cn))) {
      missing_num_++;
    }
  }

 private:
  size_t missing_num_;
};  // class LayoutChecker

Expr SetExprLayout(const Expr& func, bool allow_missing) {
  auto layout_infer = LayoutInfer();
  auto new_func = layout_infer.Infer(func);
  if (!allow_missing) {
    LayoutChecker().Check(new_func);
  }
  return new_func;
}

namespace transform {

Pass SetExprLayout(bool allow_missing) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SetExprLayout(f, allow_missing));
      };
  return CreateFunctionPass(pass_func, 1, "SetExprLayout", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SetExprLayout").set_body_typed(SetExprLayout);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
