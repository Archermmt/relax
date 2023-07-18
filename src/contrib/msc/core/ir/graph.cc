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
 * \file src/contrib/msc/core/ir/graph.cc
 */

#include "graph.h"

#include "../printer/prototxt_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

MSCTensor::MSCTensor(const String& name, const DataType& dtype, const String& layout,
                     const Array<Integer>& shape, const String& alias) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->name = std::move(name);
  n->alias = std::move(alias);
  n->dtype = std::move(dtype);
  n->shape = std::move(shape);
  n->layout = tvm::tir::Layout(layout);
  data_ = std::move(n);
}

MSCTensor::MSCTensor(const std::string& json_str) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(MSCTensorNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCTensorNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* tensor = static_cast<const MSCTensorNode*>(node.get());
      p->PrintIndent();
      p->stream << tensor->name;
      if (tensor->alias.size() > 0) {
        p->stream << "(" << tensor->alias << ")";
      }
      p->stream << "<";
      for (size_t i = 0; i < tensor->Ndim(); i++) {
        p->stream << tensor->shape[i]->value << (i == tensor->Ndim() - 1 ? "|" : ",");
      }
      p->stream << tensor->dtype;
      if (tensor->layout.defined()) {
        p->stream << "|" << tensor->layout.name();
      }
      p->stream << ">";
    });

const String MSCTensorNode::ToJson() const {
  JsonMSCTensor j_tensor;
  j_tensor.name = name;
  j_tensor.alias = alias;
  j_tensor.dtype = runtime::DLDataType2String(dtype);
  if (layout.defined()) {
    j_tensor.layout = layout.name();
  }
  for (const auto& s : shape) {
    j_tensor.shape.push_back(s->value);
  }
  return j_tensor.Export();
}

void MSCTensorNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCTensor j_tensor;
  reader.Read(&j_tensor);
  name = j_tensor.name;
  alias = j_tensor.alias;
  dtype = DataType(runtime::String2DLDataType(j_tensor.dtype));
  if (j_tensor.layout.size() > 0) {
    layout = tvm::tir::Layout(j_tensor.layout);
  }
  for (const auto& s : j_tensor.shape) {
    shape.push_back(s);
  }
}

size_t MSCTensorNode::Ndim() const { return shape.size(); }

const Integer MSCTensorNode::DimAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, Ndim());
  return shape[v_index];
}

const Integer MSCTensorNode::DimAt(const String& axis) const {
  auto index = layout.IndexOf(tvm::tir::LayoutAxis::Get(axis));
  return DimAt(index);
}

const Integer MSCTensorNode::GetSize() const {
  Integer size = Integer(1);
  for (const auto& s : shape) {
    size *= s;
  }
  return size;
}

const String MSCTensorNode::DtypeName() const { return runtime::DLDataType2String(dtype); }

#define MSC_NODE_BASE_HEAD(Stream, Joint)                     \
  Stream << "ID_" << Joint->index << " " << Joint->name;      \
  if (Joint->master.size() > 0) {                             \
    Stream << "(M: " << Joint->master << ")";                 \
  }                                                           \
  Stream << " <PARENTS: ";                                    \
  if (Joint->parents.size() > 0) {                            \
    for (size_t i = 0; i < Joint->parents.size(); i++) {      \
      Stream << Downcast<BaseJoint>(Joint->parents[i])->name  \
             << (i == Joint->parents.size() - 1 ? "" : ",");  \
    }                                                         \
  }                                                           \
  Stream << "| CHILDERN: ";                                   \
  if (Joint->children.size() > 0) {                           \
    for (size_t i = 0; i < Joint->children.size(); i++) {     \
      Stream << Downcast<BaseJoint>(Joint->children[i])->name \
             << (i == Joint->children.size() - 1 ? "" : ","); \
    }                                                         \
  }                                                           \
  Stream << ">\n";

size_t BaseJointNode::AddChild(const BaseJoint& child) const {
  for (size_t i = 0; i < children.size(); i++) {
    if (Downcast<BaseJoint>(children[i])->name == child->name) {
      return i;
    }
  }
  children.push_back(child);
  return children.size() - 1;
}

bool BaseJointNode::GetAttr(const String& key, std::string* val) const {
  if (attrs.count(key)) {
    *val = attrs[key];
    return true;
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, int* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      *val = std::stoi(val_str);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, int64_t* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      *val = std::stoi(val_str);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, float* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      *val = std::atof(val_str.c_str());
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, bool* val) const {
  int val_int;
  if (GetAttr(key, &val_int)) {
    *val = (val_int != 0);
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<int>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::stoi(std::string(s)));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<int64_t>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::stol(std::string(s)));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}
bool BaseJointNode::GetAttr(const String& key, std::vector<float>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::atof(std::string(s).c_str()));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

MSCJoint::MSCJoint(int index, const String& name, const String& master, const String& optype,
                   const Map<String, String>& attrs, const Array<String>& scope,
                   const std::vector<std::pair<BaseJoint, size_t>>& inputs,
                   const Array<MSCTensor>& outputs, const Map<String, MSCTensor>& weights) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->index = index;
  n->name = std::move(name);
  n->master = std::move(master);
  n->optype = std::move(optype);
  n->attrs = std::move(attrs);
  n->scope = std::move(scope);
  Array<ObjectRef> parents;
  Array<Array<Integer>> array_inputs;
  Array<String> added_parents;
  for (const auto& pair : inputs) {
    // const auto& parent=Downcast<BaseJoint>(pair.first);
    const auto& p_name = pair.first->name;
    int p_idx = -1;
    for (size_t i = 0; i < added_parents.size(); i++) {
      if (added_parents[i] == p_name) {
        p_idx = i;
        break;
      }
    }
    if (p_idx == -1) {
      parents.push_back(pair.first);
      added_parents.push_back(p_name);
      p_idx = added_parents.size() - 1;
    }
    Array<Integer> input{Integer(p_idx), Integer(pair.second)};
    array_inputs.push_back(input);
  }
  n->parents = std::move(parents);
  n->inputs = std::move(array_inputs);
  n->outputs = std::move(outputs);
  n->weights = std::move(weights);
  data_ = std::move(n);
}

MSCJoint::MSCJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->FromJson(json_str, nodes);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(MSCJointNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCJointNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* joint = static_cast<const MSCJointNode*>(node.get());
      p->PrintIndent();
      MSC_NODE_BASE_HEAD(p->stream, joint);
      if (joint->inputs.size() > 0) {
        p->stream << "  IN: ";
        for (size_t i = 0; i < joint->inputs.size(); i++) {
          p->stream << joint->InputAt(i) << (i == joint->inputs.size() - 1 ? "\n" : ",");
        }
      }
      p->stream << "  OUT: ";
      for (size_t i = 0; i < joint->outputs.size(); i++) {
        p->stream << joint->OutputAt(i) << (i == joint->outputs.size() - 1 ? "\n" : ",");
      }
      p->stream << "  OPTYPE: " << joint->optype << "\n";
      if (joint->scope.size() > 0) {
        p->stream << "  SCOPE: ";
        for (size_t i = 0; i < joint->scope.size(); i++) {
          p->stream << joint->scope[i] << (i == joint->scope.size() - 1 ? "\n" : ".");
        }
      }
      if (joint->attrs.size() > 0) {
        p->stream << "  ATTRS: ";
        for (const auto& pair : joint->attrs) {
          p->stream << pair.first << "=" << pair.second << " ";
        }
        p->stream << "\n";
      }
      if (joint->weights.size() > 0) {
        p->stream << "  WEIGHTS: ";
        for (const auto& pair : joint->weights) {
          p->stream << "\n    " << pair.first << ": " << pair.second;
        }
        p->stream << "\n";
      }
    });

const String MSCJointNode::ToJson() const {
  JsonMSCJoint j_joint;
  j_joint.index = index;
  j_joint.name = name;
  j_joint.master = master;
  j_joint.optype = optype;
  for (const auto& pair : attrs) {
    j_joint.attrs[pair.first] = pair.second;
  }
  for (const auto& s : scope) {
    j_joint.scope.push_back(s);
  }
  for (const auto& p : parents) {
    j_joint.parents.push_back(Downcast<BaseJoint>(p)->name);
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    j_joint.inputs.push_back(InputAt(i)->name);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    j_joint.outputs.push_back(OutputAt(i)->ToJson());
  }
  for (const auto& pair : weights) {
    j_joint.weights[pair.first] = pair.second->ToJson();
  }
  return j_joint.Export();
}

void MSCJointNode::FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCJoint j_joint;
  reader.Read(&j_joint);
  index = j_joint.index;
  name = j_joint.name;
  master = j_joint.master;
  optype = j_joint.optype;
  for (const auto& pair : j_joint.attrs) {
    attrs.Set(pair.first, pair.second);
  }
  for (const auto& s : j_joint.scope) {
    scope.push_back(s);
  }
  for (const auto& p_name : j_joint.parents) {
    ICHECK(nodes.count(p_name)) << "Can not find parent " << p_name;
    parents.push_back(nodes[p_name]);
  }
  for (const auto& in_name : j_joint.inputs) {
    String producer, index_str;
    std::tie(producer, index_str) = StringUtils::SplitOnce(in_name, ":");
    int p_idx = -1;
    for (size_t i = 0; i < parents.size(); i++) {
      if (Downcast<BaseJoint>(parents[i])->name == producer) {
        p_idx = i;
        break;
      }
    }
    ICHECK(p_idx >= 0) << "Can not find parent for " << in_name;
    Array<Integer> input{Integer(p_idx), Integer(std::stol(index_str))};
    inputs.push_back(input);
  }
  for (const auto& o : j_joint.outputs) {
    outputs.push_back(MSCTensor(o));
  }
  for (const auto& pair : j_joint.weights) {
    weights.Set(pair.first, MSCTensor(pair.second));
  }
}

const MSCTensor MSCJointNode::InputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, inputs.size());
  const auto& p_idx = inputs[v_index][0];
  const auto& out_idx = inputs[v_index][1];
  return Downcast<MSCJoint>(parents[p_idx->value])->OutputAt(out_idx->value);
}

const MSCTensor MSCJointNode::OutputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, outputs.size());
  return outputs[v_index];
}

const MSCTensor MSCJointNode::WeightAt(const String& wtype) const {
  ICHECK(weights.count(wtype)) << "Can not find " << wtype << " from weights";
  return weights[wtype];
}

const MSCJoint MSCJointNode::ProducerOf(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, inputs.size());
  const auto& p_idx = inputs[v_index][0];
  return Downcast<MSCJoint>(parents[p_idx->value]);
}

const MSCJoint MSCJointNode::ProducerOf(const String& input_name) const {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (InputAt(i)->name == input_name) {
      return ProducerOf(i);
    }
  }
  return MSCJoint();
}

const MSCJoint MSCJointNode::ProducerOf(const MSCTensor& input) const {
  return ProducerOf(input->name);
}

WeightJoint::WeightJoint(int index, const String& name, const String& master, const String& optype,
                         const String& wtype, const Map<String, String>& attrs,
                         const MSCTensor& weight, const Array<BaseJoint> parents,
                         const Array<BaseJoint>& friends) {
  ObjectPtr<WeightJointNode> n = make_object<WeightJointNode>();
  n->index = index;
  n->name = std::move(name);
  n->master = std::move(master);
  n->optype = std::move(optype);
  n->wtype = std::move(wtype);
  n->attrs = std::move(attrs);
  n->weight = std::move(weight);
  for (const auto& p : parents) {
    n->parents.push_back(p);
  }
  n->friends = std::move(friends);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(WeightJointNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WeightJointNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* joint = static_cast<const WeightJointNode*>(node.get());
      p->PrintIndent();
      MSC_NODE_BASE_HEAD(p->stream, joint);
      if (joint->friends.size() > 0) {
        p->stream << "  FRIENDS: ";
        for (size_t i = 0; i < joint->friends.size(); i++) {
          p->stream << Downcast<BaseJoint>(joint->friends[i])->name
                    << (i == joint->friends.size() - 1 ? "\n" : ",");
        }
      }
      p->stream << "  OPTYPE: " << joint->optype;
      p->stream << "\n  WEIGHT_TYPE: " << joint->wtype;
      p->stream << "\n  WEIGHT: " << joint->weight;
      if (joint->attrs.size() > 0) {
        p->stream << "\n  ATTRS: ";
        for (const auto& pair : joint->attrs) {
          p->stream << pair.first << " = " << pair.second << " ";
        }
      }
      p->stream << "\n";
    });

MSCGraph::MSCGraph(const String& name, const Array<MSCJoint>& nodes,
                   const Array<String>& input_names, const Array<String>& output_names) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->name = std::move(name);
  for (const auto& node : nodes) {
    n->node_names.push_back(node->name);
    n->nodes.Set(node->name, node);
  }
  n->input_names = std::move(input_names);
  n->output_names = std::move(output_names);
  n->AnalysisGraph();
  data_ = std::move(n);
}

MSCGraph::MSCGraph(const std::string& json_str) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(MSCGraphNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCGraphNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* graph = static_cast<const MSCGraphNode*>(node.get());
      p->PrintIndent();
      p->stream << graph->name << " <INPUTS: ";
      for (size_t i = 0; i < graph->input_names.size(); i++) {
        p->stream << graph->input_names[i] << (i == graph->input_names.size() - 1 ? "| " : ",");
      }
      p->stream << "OUTPUTS: ";
      for (size_t i = 0; i < graph->output_names.size(); i++) {
        p->stream << graph->output_names[i] << (i == graph->output_names.size() - 1 ? "\n" : ",");
      }
      for (const auto& n : graph->node_names) {
        p->stream << graph->FindNode(n) << "\n";
      }
    });

const String MSCGraphNode::ToJson() const {
  JsonMSCGarph j_graph;
  j_graph.name = name;
  for (const auto& i : input_names) {
    j_graph.inputs.push_back(i);
  }
  for (const auto& o : output_names) {
    j_graph.outputs.push_back(o);
  }
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    j_graph.nodes.push_back(node->ToJson());
  }
  return j_graph.Export();
}

void MSCGraphNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCGarph j_graph;
  reader.Read(&j_graph);
  name = j_graph.name;
  for (const auto& i : j_graph.inputs) {
    input_names.push_back(i);
  }
  for (const auto& o : j_graph.outputs) {
    output_names.push_back(o);
  }
  Map<String, BaseJoint> loaded_nodes;
  for (const auto& n : j_graph.nodes) {
    const auto& node = MSCJoint(n, loaded_nodes);
    loaded_nodes.Set(node->name, node);
    for (size_t i = 0; i < node->parents.size(); i++) {
      Downcast<BaseJoint>(node->parents[i])->AddChild(node);
    }
    node_names.push_back(node->name);
    nodes.Set(node->name, node);
  }
  AnalysisGraph();
}

const String MSCGraphNode::ToPrototxt() const {
  PrototxtPrinter printer;
  printer.Append(Map<String, ObjectRef>{{"name", name}});
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    // define layer
    std::vector<std::pair<String, ObjectRef>> layer;
    layer.push_back(std::make_pair("name", node->name));
    layer.push_back(std::make_pair("type", StringUtils::Replace(node->optype, ".", "_")));
    layer.push_back(std::make_pair("top", node->name));
    for (const auto& p : node->parents) {
      layer.push_back(std::make_pair("bottom", Downcast<BaseJoint>(p)->name));
    }
    // define layer param
    Map<String, ObjectRef> param;
    param.Set("idx", Integer(node->index));
    for (size_t i = 0; i < node->inputs.size(); i++) {
      param.Set("input_" + std::to_string(i), node->InputAt(i));
    }
    for (size_t i = 0; i < node->outputs.size(); i++) {
      param.Set("output_" + std::to_string(i), node->OutputAt(i));
    }
    for (const auto& pair : node->weights) {
      param.Set("w_" + pair.first, pair.second);
    }
    layer.push_back(std::make_pair("layer_param", PrototxtPrinter::ToDictDoc(param)));
    // Append the layer Map
    printer.Append(Map<String, ObjectRef>{{"layer", PrototxtPrinter::ToDictDoc(layer)}});
  }
  return printer.GetString();
}

const MSCJoint MSCGraphNode::FindNode(const String& name) const {
  ICHECK(nodes.count(name)) << "Can not find node " << name;
  return Downcast<MSCJoint>(nodes[name]);
}

const MSCTensor MSCGraphNode::InputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, input_names.size());
  return FindTensor(input_names[v_index]);
}

const MSCTensor MSCGraphNode::OutputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, output_names.size());
  return FindTensor(output_names[v_index]);
}

const MSCTensor MSCGraphNode::FindTensor(const String& name) const {
  if (weight_holders.count(name)) {
    const auto& node = FindNode(weight_holders[name][0]);
    for (const auto& pair : node->weights) {
      if (pair.second->name == name) {
        return pair.second;
      }
    }
    LOG(FATAL) << "Can not find weight " << name << " from " << node;
  }
  const auto& pair = FindProducerAndIdx(name);
  return pair.first->OutputAt(pair.second);
}

const MSCJoint MSCGraphNode::FindProducer(const String& name) const {
  const auto& pair = FindProducerAndIdx(name);
  return pair.first;
}

const MSCJoint MSCGraphNode::FindProducer(const MSCTensor& tensor) const {
  return FindProducer(tensor->name);
}

const std::pair<MSCJoint, size_t> MSCGraphNode::FindProducerAndIdx(const String& name) const {
  ICHECK(!weight_holders.count(name)) << "Weight has no producer";
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  String host, index;
  std::tie(host, index) = StringUtils::SplitOnce(tensor_name, ":");
  return std::make_pair(FindNode(host), std::stoi(index));
}

const std::pair<MSCJoint, size_t> MSCGraphNode::FindProducerAndIdx(const MSCTensor& tensor) const {
  return FindProducerAndIdx(tensor->name);
}

const Array<MSCJoint> MSCGraphNode::FindConsumers(const String& name) const {
  Array<MSCJoint> consumers;
  if (weight_holders.count(name)) {
    for (const auto& h : weight_holders[name]) {
      consumers.push_back(FindNode(h));
    }
  } else {
    const auto& producer = FindProducer(name);
    for (const auto& c : producer->children) {
      consumers.push_back(Downcast<MSCJoint>(c));
    }
  }
  return consumers;
}

const Array<MSCJoint> MSCGraphNode::FindConsumers(const MSCTensor& tensor) const {
  return FindConsumers(tensor->name);
}

const std::vector<std::pair<MSCJoint, size_t>> MSCGraphNode::FindConsumersAndIndices(
    const String& name) const {
  ICHECK(!weight_holders.count(name)) << "Weight has no index";
  std::vector<std::pair<MSCJoint, size_t>> consumers;
  for (const auto& c : FindConsumers(name)) {
    bool find_tensor = false;
    for (size_t i = 0; i < c->inputs.size(); i++) {
      if (c->InputAt(i)->name == name) {
        consumers.push_back(std::make_pair(c, i));
        find_tensor = true;
        break;
      }
    }
    ICHECK(find_tensor) << "Can not find tensor " << name << " from " << c;
  }
  return consumers;
}

const std::vector<std::pair<MSCJoint, size_t>> MSCGraphNode::FindConsumersAndIndices(
    const MSCTensor& tensor) const {
  return FindConsumersAndIndices(tensor->name);
}

void MSCGraphNode::AnalysisGraph() {
  // Add children
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    for (const auto& p : node->parents) {
      Downcast<MSCJoint>(p)->AddChild(node);
    }
  }
  // Check inputs and outputs
  for (const auto& i : input_names) {
    const auto& input = FindTensor(i);
    if (input->alias.size() > 0) {
      tensor_alias.Set(input->alias, input->name);
    }
  }
  for (const auto& o : output_names) {
    FindTensor(o);
  }
  // Set tensor alias and weight_holders
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    for (const auto& o : node->outputs) {
      if (o->alias.size() > 0) {
        tensor_alias.Set(o->alias, o->name);
      }
    }
    for (const auto& pair : node->weights) {
      const auto& w_name = pair.second->name;
      if (weight_holders.count(w_name)) {
        Array<String> holders = weight_holders[w_name];
        holders.push_back(n);
        weight_holders.Set(w_name, holders);
      } else {
        weight_holders.Set(w_name, Array<String>({n}));
      }
    }
  }
}

WeightGraph::WeightGraph(const String& name, const Array<WeightJoint>& nodes) {
  ObjectPtr<WeightGraphNode> n = make_object<WeightGraphNode>();
  n->name = std::move(name);
  for (const auto& node : nodes) {
    n->node_names.push_back(node->name);
    n->nodes.Set(node->name, node);
  }
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(WeightGraphNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WeightGraphNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* graph = static_cast<const WeightGraphNode*>(node.get());
      p->PrintIndent();
      p->stream << graph->name << "\n";
      for (const auto& n : graph->node_names) {
        p->stream << graph->FindNode(n) << "\n";
      }
    });

const String WeightGraphNode::ToPrototxt() const {
  PrototxtPrinter printer;
  printer.Append(Map<String, ObjectRef>{{"name", name}});
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    // define layer
    std::vector<std::pair<String, ObjectRef>> layer;
    layer.push_back(std::make_pair("name", node->name));
    layer.push_back(
        std::make_pair("type", StringUtils::Replace(node->optype, ".", "_") + "_" + node->wtype));
    layer.push_back(std::make_pair("top", node->name));
    for (const auto& p : node->parents) {
      layer.push_back(std::make_pair("bottom", Downcast<BaseJoint>(p)->name));
    }
    // define layer param
    Map<String, ObjectRef> param;
    param.Set("idx", Integer(node->index));
    param.Set("weight", node->weight);
    for (size_t i = 0; i < node->friends.size(); i++) {
      param.Set("friend_" + std::to_string(i), Downcast<MSCJoint>(node->friends[i]));
    }
    layer.push_back(std::make_pair("layer_param", PrototxtPrinter::ToDictDoc(param)));
    // Append the layer Map
    printer.Append(Map<String, ObjectRef>{{"layer", PrototxtPrinter::ToDictDoc(layer)}});
  }
  return printer.GetString();
}

const WeightJoint WeightGraphNode::FindNode(const String& name) const {
  ICHECK(nodes.count(name)) << "Can not find node " << name;
  return Downcast<WeightJoint>(nodes[name]);
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm