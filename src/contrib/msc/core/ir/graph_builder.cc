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
 * \file src/contrib/msc/core/ir/graph_builder.cc
 */

#include "graph_builder.h"

namespace tvm {
namespace contrib {
namespace msc {

MSCGraph RelaxGraphBuilder::Build(const relax::Function& func) {
  // Add input nodes and record inputs;
  Array<String> input_names, output_names;
  for (const auto& p : func->params) {
    AddNode(p, NullOpt, p->name_hint());
    ICHECK(expr_tensor_map_.count(p)) << "Can not find func param " << p;
    input_names.push_back(expr_tensor_map_[p][0]);
  }
  VisitExpr(func);
  if (const auto* b_node = func->body.as<relax::SeqExprNode>()) {
    ICHECK(expr_tensor_map_.count(b_node->body)) << "Can not find seqexpr body " << b_node->body;
    output_names = expr_tensor_map_[b_node->body];
  } else {
    LOG(FATAL) << "Function body should be SeqExpr, get " << func->body;
  }
  // remove const nodes as weights
  Array<MSCJoint> valid_nodes;
  for (const auto& n : nodes_) {
    if (!weights_.count(n->name)) {
      n->index = valid_nodes.size();
      valid_nodes.push_back(n);
    }
  }
  const auto& graph = MSCGraph(name_, valid_nodes, input_names, output_names);
  // set inputs and outputs alias
  if (config_.input_aliass.size() == input_names.size()) {
    for (size_t i = 0; i < input_names.size(); i++) {
      graph->FindTensor(input_names[i])->alias = config_.input_aliass[i];
    }
  }
  if (config_.output_aliass.size() == output_names.size()) {
    for (size_t i = 0; i < output_names.size(); i++) {
      graph->FindTensor(output_names[i])->alias = config_.output_aliass[i];
    }
  }
  return graph;
}

MSCJoint RelaxGraphBuilder::AddNode(const Expr& expr, const Optional<Expr>& binding_var,
                                    const String& name) {
  const auto& node_name = name.size() > 0 ? name : SpanUtils::GetAttr(expr->span, "name");
  const auto& master_name = SpanUtils::GetAttr(expr->span, "master");
  String optype;
  if (const auto* var = expr.as<relax::VarNode>()) {
    optype = var->name_hint();
  } else if (expr.as<relax::ConstantNode>()) {
    optype = "constant";
  } else if (const auto* call_node = expr.as<relax::CallNode>()) {
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      optype = op_node->name;
    } else if (call_node->op.as<relax::FunctionNode>()) {
      optype = "function";
    } else if (call_node->op.as<GlobalVarNode>()) {
      optype = "global_function";
    } else {
      optype = "unknown_op";
    }
  } else {
    optype = "unknown_expr";
  }
  // Extract attributes
  Map<String, String> attrs;
  if (const auto* call_node = expr.as<relax::CallNode>()) {
    if (call_node->attrs.defined()) {
      Array<String> keys, values;
      AttrGetter getter(&keys, &values);
      const_cast<BaseAttrsNode*>(call_node->attrs.get())->VisitAttrs(&getter);
      for (size_t i = 0; i < keys.size(); i++) {
        attrs.Set(keys[i], values[i]);
      }
    }
  }
  // Get scope
  const auto& scope = StringUtils::Split(scope_name_, ".");
  // Build inputs and weights
  std::vector<std::pair<BaseJoint, size_t>> inputs;
  Map<String, MSCTensor> node_weights;
  if (const auto* call_node = expr.as<relax::CallNode>()) {
    Array<String> input_names;
    const auto& input_types = ExprUtils::GetInputTypes(Downcast<relax::Call>(expr));
    for (size_t i = 0; i < call_node->args.size(); i++) {
      const auto& arg = call_node->args[i];
      if (const auto* s_node = arg.as<relax::ShapeExprNode>()) {
        attrs.Set("shape", StringUtils::ToString(s_node->values));
        continue;
      }
      ICHECK(expr_tensor_map_.count(arg)) << "Missing argument " << arg;
      if (input_types[i] != "input" && arg.as<relax::ConstantNode>()) {
        const auto& t_name = expr_tensor_map_[arg][0];
        const auto& w_name = SpanUtils::GetAttr(arg->span, "name");
        if (!weights_.count(w_name)) {
          const auto& pair = tensor_input_map_[t_name];
          const auto& ref = Downcast<MSCJoint>(pair.first)->OutputAt(pair.second);
          const auto& weight = MSCTensor(w_name, ref->dtype, ref->layout.name(), ref->shape);
          weights_.Set(w_name, weight);
        }
        node_weights.Set(input_types[i], weights_[w_name]);
      } else {
        for (const auto& in_name : expr_tensor_map_[arg]) {
          input_names.push_back(in_name);
        }
      }
    }
    for (const auto& i : input_names) {
      inputs.push_back(tensor_input_map_[i]);
    }
  }
  // Build outputs
  Array<MSCTensor> outputs;
  const auto& layout = SpanUtils::GetAttr(expr->span, "layout");
  const auto& sinfo = relax::GetStructInfo(expr);
  if (const auto* t_info = sinfo.as<relax::TensorStructInfoNode>()) {
    const auto& opt_shape = t_info->GetShape();
    Array<Integer> shape;
    if (opt_shape.defined()) {
      for (const auto& s : opt_shape.value()) {
        shape.push_back(Downcast<Integer>(s));
      }
    }
    const auto& output =
        MSCTensor(node_name + ":" + std::to_string(0), t_info->dtype, layout, shape);
    outputs.push_back(output);
  } else if (const auto* tuple_sinfo = sinfo.as<relax::TupleStructInfoNode>()) {
    Array<String> layouts = StringUtils::Split(layout, ",");
    if (layouts.size() == 0) {
      layouts = Array<String>(tuple_sinfo->fields.size(), "");
    }
    ICHECK_EQ(layouts.size(), tuple_sinfo->fields.size())
        << "Layout " << layout << " msimatch with fileds size " << tuple_sinfo->fields.size();
    for (size_t i = 0; i < tuple_sinfo->fields.size(); i++) {
      const auto& t_info = Downcast<relax::TensorStructInfo>(tuple_sinfo->fields[i]);
      const auto& opt_shape = t_info->GetShape();
      Array<Integer> shape;
      if (opt_shape.defined()) {
        for (const auto& s : opt_shape.value()) {
          shape.push_back(Downcast<Integer>(s));
        }
      }
      const auto& output =
          MSCTensor(node_name + ":" + std::to_string(i), t_info->dtype, layouts[i], shape);
      outputs.push_back(output);
    }
  } else {
    LOG(FATAL) << "Unexpected struct info " << sinfo;
  }
  // Build node
  const auto& node = MSCJoint(nodes_.size(), node_name, master_name, optype, attrs, scope, inputs,
                              outputs, node_weights);
  Array<String> output_names;
  for (size_t i = 0; i < outputs.size(); i++) {
    output_names.push_back(outputs[i]->name);
    tensor_input_map_[outputs[i]->name] = std::make_pair(node, i);
  }
  nodes_.push_back(node);
  const auto& ref_expr = binding_var.defined() ? binding_var.value() : expr;
  expr_tensor_map_.Set(ref_expr, output_names);
  return node;
}

void RelaxGraphBuilder::VisitBindingBlock(const relax::BindingBlock& block) {
  scope_name_ = SpanUtils::GetAttr(block->span, "name");
  RelaxExprVisitor::VisitBindingBlock(block);
}

void RelaxGraphBuilder::VisitExpr_(const relax::ConstantNode* op) {
  AddNode(GetRef<relax::Constant>(op));
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::CallNode* call_node) {
  RelaxExprVisitor::VisitBinding_(binding, call_node);
  AddNode(GetRef<relax::Call>(call_node), binding->var);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::TupleNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  Array<String> tensors;
  for (const auto& f : val->fields) {
    ICHECK(expr_tensor_map_.count(f)) << "Can not find tuple field " << f;
    for (const auto& n : expr_tensor_map_[f]) {
      tensors.push_back(n);
    }
  }
  expr_tensor_map_.Set(binding->var, tensors);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::TupleGetItemNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  Array<String> tensors;
  ICHECK(expr_tensor_map_.count(val->tuple)) << "Can not find tuple " << val->tuple;
  const auto& tensor_names = expr_tensor_map_[val->tuple];
  ICHECK_LT(val->index, tensor_names.size())
      << "Index " << val->index << " out of range " << tensor_names;
  expr_tensor_map_.Set(binding->var, Array<String>{tensor_names[val->index]});
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::DataflowVarNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  const auto& output = GetRef<relax::DataflowVar>(val);
  ICHECK(expr_tensor_map_.count(output)) << "Can not find output " << output;
  expr_tensor_map_.Set(binding->var, expr_tensor_map_[output]);
}

void RelaxWeightsExtractor::VisitExpr_(const relax::ConstantNode* op) {
  weights_.Set(SpanUtils::GetAttr(op->span, "name"), op->data);
}

Map<String, NDArray> RelaxWeightsExtractor::GetWeights(const relax::Function& func) {
  VisitExpr(func);
  return weights_;
}

MSCGraph BuildFromRelax(const IRModule& relax_module, const String& entry_name,
                        const String& options) {
  const auto& func = Downcast<relax::Function>(relax_module->Lookup(entry_name));
  return RelaxGraphBuilder(entry_name, options).Build(func);
}

Map<String, NDArray> GetRelaxWeights(const IRModule& relax_module, const String& entry_name) {
  const auto& func = Downcast<relax::Function>(relax_module->Lookup(entry_name));
  return RelaxWeightsExtractor().GetWeights(func);
}

TVM_REGISTER_GLOBAL("msc.core.BuildFromRelax").set_body_typed(BuildFromRelax);
TVM_REGISTER_GLOBAL("msc.core.GetRelaxWeights").set_body_typed(GetRelaxWeights);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm