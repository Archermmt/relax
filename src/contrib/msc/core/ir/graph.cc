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

#include <set>

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

MSCTensor::MSCTensor(const JsonMSCTensor& j_tensor) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->FromJson(j_tensor);
  data_ = std::move(n);
}

MSCTensor::MSCTensor(const std::string& json_str) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonMSCTensor MSCTensorNode::ToJson() const {
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
  return j_tensor;
}

void MSCTensorNode::FromJson(const JsonMSCTensor& j_tensor) {
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

void MSCTensorNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCTensor j_tensor;
  reader.Read(&j_tensor);
  FromJson(j_tensor);
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

const String MSCTensorNode::DTypeName() const { return runtime::DLDataType2String(dtype); }

size_t BaseJointNode::AddChild(const BaseJoint& child) const {
  for (size_t i = 0; i < children.size(); i++) {
    if (Downcast<BaseJoint>(children[i])->name == child->name) {
      return i;
    }
  }
  children.push_back(child);
  return children.size() - 1;
}

const BaseJoint BaseJointNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<BaseJoint>(parents[v_index]);
}

const BaseJoint BaseJointNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<BaseJoint>(children[v_index]);
}

bool BaseJointNode::HasAttr(const String& key) const { return attrs.count(key); }

bool BaseJointNode::GetAttr(const String& key, std::string* val) const {
  if (attrs.count(key) && attrs[key].size() > 0) {
    *val = attrs[key];
    return true;
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, int* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos > 0) {
      return false;
    }
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
    return true;
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<int>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos < 0) {
      return false;
    }
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
      int pos = val_str.find(",");
      if (pos < 0) {
        return false;
      }
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
    int pos = val_str.find(",");
    if (pos < 0) {
      return false;
    }
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

MSCJoint::MSCJoint(const JsonMSCJoint& j_joint, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->FromJson(j_joint, nodes);
  data_ = std::move(n);
}

MSCJoint::MSCJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->FromJson(json_str, nodes);
  data_ = std::move(n);
}

const MSCJoint MSCJoint::Clone(const MSCJoint& node,
                               const std::vector<std::pair<BaseJoint, size_t>>& inputs) {
  return MSCJoint(node->index, node->name, node->master, node->optype, node->attrs, node->scope,
                  inputs, node->outputs, node->weights);
}

const JsonMSCJoint MSCJointNode::ToJson() const {
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
  for (const auto& i : GetInputs()) {
    j_joint.inputs.push_back(i->name);
  }
  for (const auto& o : GetOutputs()) {
    j_joint.outputs.push_back(o->ToJson());
  }
  for (const auto& pair : weights) {
    j_joint.weights[pair.first] = pair.second->ToJson();
  }
  return j_joint;
}

void MSCJointNode::FromJson(const JsonMSCJoint& j_joint, const Map<String, BaseJoint>& nodes) {
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
      if (ParentAt(i)->name == producer) {
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

void MSCJointNode::FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCJoint j_joint;
  reader.Read(&j_joint);
  FromJson(j_joint, nodes);
}

const MSCTensor MSCJointNode::InputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, inputs.size());
  const auto& p_idx = inputs[v_index][0];
  const auto& out_idx = inputs[v_index][1];
  return ParentAt(p_idx->value)->OutputAt(out_idx->value);
}

const Array<MSCTensor> MSCJointNode::GetInputs() const {
  Array<MSCTensor> t_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    t_inputs.push_back(InputAt(i));
  }
  return t_inputs;
}

const MSCTensor MSCJointNode::OutputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, outputs.size());
  return outputs[v_index];
}

const Array<MSCTensor> MSCJointNode::GetOutputs() const {
  Array<MSCTensor> t_outputs;
  for (size_t i = 0; i < outputs.size(); i++) {
    t_outputs.push_back(OutputAt(i));
  }
  return t_outputs;
}

const MSCTensor MSCJointNode::WeightAt(const String& wtype) const {
  ICHECK(weights.count(wtype)) << "Can not find " << wtype << " from weights";
  return weights[wtype];
}

const MSCJoint MSCJointNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<MSCJoint>(parents[v_index]);
}

const MSCJoint MSCJointNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<MSCJoint>(children[v_index]);
}

const MSCJoint MSCJointNode::ProducerOf(int index) const {
  const auto& pair = ProducerAndIdxOf(index);
  return pair.first;
}

const MSCJoint MSCJointNode::ProducerOf(const String& input_name) const {
  const auto& pair = ProducerAndIdxOf(input_name);
  return pair.first;
}

const MSCJoint MSCJointNode::ProducerOf(const MSCTensor& input) const {
  return ProducerOf(input->name);
}

const std::pair<MSCJoint, size_t> MSCJointNode::ProducerAndIdxOf(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, inputs.size());
  const auto& p_idx = inputs[v_index][0];
  return std::make_pair(ParentAt(p_idx->value), inputs[v_index][1]->value);
}

const std::pair<MSCJoint, size_t> MSCJointNode::ProducerAndIdxOf(const String& name) const {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (InputAt(i)->name == name) {
      return ProducerAndIdxOf(i);
    }
  }
  LOG(FATAL) << "Can not find producer of " << name;
}

const std::pair<MSCJoint, size_t> MSCJointNode::ProducerAndIdxOf(const MSCTensor& input) const {
  return ProducerAndIdxOf(input->name);
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

const WeightJoint WeightJointNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<WeightJoint>(parents[v_index]);
}

const WeightJoint WeightJointNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<WeightJoint>(children[v_index]);
}

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

MSCGraph::MSCGraph(const JsonMSCGraph& j_graph) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->FromJson(j_graph);
  data_ = std::move(n);
}

MSCGraph::MSCGraph(const std::string& json_str) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonMSCGraph MSCGraphNode::ToJson() const {
  JsonMSCGraph j_graph;
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
  return j_graph;
}

void MSCGraphNode::FromJson(const JsonMSCGraph& j_graph) {
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
    for (const auto& p : node->parents) {
      Downcast<BaseJoint>(p)->AddChild(node);
    }
    node_names.push_back(node->name);
    nodes.Set(node->name, node);
  }
  AnalysisGraph();
}

void MSCGraphNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCGraph j_graph;
  reader.Read(&j_graph);
  FromJson(j_graph);
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
      param.Set("param_" + pair.first, pair.second);
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

const Array<MSCTensor> MSCGraphNode::GetInputs() const {
  Array<MSCTensor> t_inputs;
  for (size_t i = 0; i < input_names.size(); i++) {
    t_inputs.push_back(InputAt(i));
  }
  return t_inputs;
}

const MSCTensor MSCGraphNode::OutputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, output_names.size());
  return FindTensor(output_names[v_index]);
}

const Array<MSCTensor> MSCGraphNode::GetOutputs() const {
  Array<MSCTensor> t_outputs;
  for (size_t i = 0; i < output_names.size(); i++) {
    t_outputs.push_back(OutputAt(i));
  }
  return t_outputs;
}

const Array<MSCJoint> MSCGraphNode::GetEntries() const {
  Array<MSCJoint> entries;
  for (size_t i = 0; i < input_names.size(); i++) {
    entries.push_back(FindProducer(input_names[i]));
  }
  return entries;
}

const Array<MSCJoint> MSCGraphNode::GetExits() const {
  Array<MSCJoint> exits;
  std::set<String> setted_exits;
  for (size_t i = 0; i < output_names.size(); i++) {
    const auto& exit = FindProducer(output_names[i]);
    if (setted_exits.count(exit->name)) {
      continue;
    }
    exits.push_back(exit);
    setted_exits.insert(exit->name);
  }
  return exits;
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
  if (index.size() == 0) {
    const auto& node = FindNode(host);
    ICHECK(node->optype == "constant") << "Tensor without index should be constant, get " << node;
    return std::make_pair(node, 0);
  }
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

#define MSC_NODE_BASE_HEAD(Stream, Joint)                                                \
  Stream << "ID_" << Joint->index << " " << Joint->name;                                 \
  if (Joint->master.size() > 0) {                                                        \
    Stream << "(M: " << Joint->master << ")";                                            \
  }                                                                                      \
  Stream << " <PARENTS: ";                                                               \
  if (Joint->parents.size() > 0) {                                                       \
    for (size_t i = 0; i < Joint->parents.size(); i++) {                                 \
      Stream << Joint->ParentAt(i)->name << (i == Joint->parents.size() - 1 ? "" : ","); \
    }                                                                                    \
  }                                                                                      \
  Stream << "| CHILDERN: ";                                                              \
  if (Joint->children.size() > 0) {                                                      \
    for (size_t i = 0; i < Joint->children.size(); i++) {                                \
      Stream << Joint->ChildAt(i)->name << (i == Joint->children.size() - 1 ? "" : ","); \
    }                                                                                    \
  }                                                                                      \
  Stream << ">\n";

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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WeightGraphNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* graph = static_cast<const WeightGraphNode*>(node.get());
      p->PrintIndent();
      p->stream << graph->name << "\n";
      for (const auto& n : graph->node_names) {
        p->stream << graph->FindNode(n) << "\n";
      }
    });

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
        p->stream << graph->output_names[i] << (i == graph->output_names.size() - 1 ? ">\n" : ",");
      }
      for (const auto& n : graph->node_names) {
        p->stream << graph->FindNode(n) << "\n";
      }
    });

TVM_REGISTER_NODE_TYPE(MSCTensorNode);

TVM_REGISTER_NODE_TYPE(MSCJointNode);

TVM_REGISTER_NODE_TYPE(WeightJointNode);

TVM_REGISTER_NODE_TYPE(MSCGraphNode);

TVM_REGISTER_NODE_TYPE(WeightGraphNode);

TVM_REGISTER_GLOBAL("msc.core.MSCTensor")
    .set_body_typed([](const String& name, const DataType& dtype, const String& layout,
                       const Array<Integer>& shape, const String& alias) -> MSCTensor {
      return MSCTensor(name, dtype, layout, shape, alias);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJoint")
    .set_body_typed([](Integer index, const String& name, const String& master,
                       const String& optype, const Map<String, String>& attrs,
                       const Array<String>& scope, const Array<MSCJoint>& parents,
                       const Array<Integer> out_indices, const Array<MSCTensor>& outputs,
                       const Map<String, MSCTensor>& weights) -> MSCJoint {
      std::vector<std::pair<BaseJoint, size_t>> inputs;
      for (size_t i = 0; i < parents.size(); i++) {
        inputs.push_back(std::make_pair(parents[i], out_indices[i]->value));
      }
      return MSCJoint(index->value, name, master, optype, attrs, scope, inputs, outputs, weights);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightJoint")
    .set_body_typed([](Integer index, const String& name, const String& master,
                       const String& optype, const String& wtype, const Map<String, String>& attrs,
                       const MSCTensor& weight, const Array<WeightJoint> parents,
                       const Array<WeightJoint>& friends) -> WeightJoint {
      Array<BaseJoint> b_parents, b_friends;
      for (const auto& p : parents) {
        b_parents.push_back(p);
      }
      for (const auto& f : friends) {
        b_friends.push_back(f);
      }
      return WeightJoint(index->value, name, master, optype, wtype, attrs, weight, b_parents,
                         b_friends);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraph")
    .set_body_typed([](const String& name, const Array<MSCJoint>& nodes,
                       const Array<String>& input_names,
                       const Array<String>& output_names) -> MSCGraph {
      return MSCGraph(name, nodes, input_names, output_names);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightGraph")
    .set_body_typed([](const String& name, const Array<WeightJoint>& nodes) -> WeightGraph {
      return WeightGraph(name, nodes);
    });

// Graph APIS
TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindNode")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCJoint {
      return graph->FindNode(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindTensor")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCTensor {
      return graph->FindTensor(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindProducer")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCJoint {
      return graph->FindProducer(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindConsumers")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> Array<MSCJoint> {
      return graph->FindConsumers(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphInputAt")
    .set_body_typed([](const MSCGraph& graph, int index) -> MSCTensor {
      return graph->InputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphOutputAt")
    .set_body_typed([](const MSCGraph& graph, int index) -> MSCTensor {
      return graph->OutputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphGetInputs")
    .set_body_typed([](const MSCGraph& graph) -> Array<MSCTensor> { return graph->GetInputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphGetOutputs")
    .set_body_typed([](const MSCGraph& graph) -> Array<MSCTensor> { return graph->GetOutputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphToJson").set_body_typed([](const MSCGraph& graph) -> String {
  const auto& graph_json = graph->ToJson();
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  graph_json.Save(&writer);
  return os.str();
});

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFromJson")
    .set_body_typed([](const String& graph_json) -> MSCGraph { return MSCGraph(graph_json); });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphToPrototxt")
    .set_body_typed([](const MSCGraph& graph) -> String { return graph->ToPrototxt(); });

TVM_REGISTER_GLOBAL("msc.core.MSCJointInputAt")
    .set_body_typed([](const MSCJoint& node, int index) -> MSCTensor {
      return node->InputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJointOutputAt")
    .set_body_typed([](const MSCJoint& node, int index) -> MSCTensor {
      return node->OutputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJointGetInputs")
    .set_body_typed([](const MSCJoint& node) -> Array<MSCTensor> { return node->GetInputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCJointGetOutputs")
    .set_body_typed([](const MSCJoint& node) -> Array<MSCTensor> { return node->GetOutputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorDTypeName")
    .set_body_typed([](const MSCTensor& tensor) -> String { return tensor->DTypeName(); });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
