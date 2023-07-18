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
 * \file src/contrib/msc/core/ir/graph.h
 * \brief Core MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_IR_MSC_GRAPH_H_
#define TVM_CONTRIB_MSC_CORE_IR_MSC_GRAPH_H_

#include <dmlc/json.h>
#include <tvm/tir/data_layout.h>

#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief Json serialize and deserialize for MSCTensor.
 *  MSCTensor is edge in MSCGraph with name, dtype and shape
 */
struct JsonMSCTensor {
  std::string name;
  std::string alias;
  std::string dtype;
  std::string layout;
  std::vector<int64_t> shape;

  const std::string Export() const {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    Save(&writer);
    writer.EndObject();
    return os.str();
  }

  void Save(dmlc::JSONWriter* writer) const {
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("alias", alias);
    writer->WriteObjectKeyValue("dtype", dtype);
    writer->WriteObjectKeyValue("layout", layout);
    writer->WriteObjectKeyValue("shape", shape);
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "name") {
        reader->Read(&name);
        bitmask |= 1;
      } else if (key == "alias") {
        reader->Read(&alias);
      } else if (key == "dtype") {
        reader->Read(&dtype);
        bitmask |= 2;
      } else if (key == "layout") {
        reader->Read(&layout);
      } else if (key == "shape") {
        reader->Read(&shape);
        bitmask |= 4;
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4) << "name, dtype and shape should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for MSCJoint.
 *  MSCJoint is node in MSCGraph with name, optype and attrbutes.
 *  MSCJoint has MSCTensors as inputs, outputs and weights.
 */
struct JsonMSCJoint {
  size_t index;
  std::string name;
  std::string master;
  std::string optype;
  std::vector<std::string> scope;
  std::vector<std::string> parents;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, std::string> attrs;
  std::unordered_map<std::string, std::string> weights;

  const std::string Export() const {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    Save(&writer);
    writer.EndObject();
    return os.str();
  }

  void Save(dmlc::JSONWriter* writer) const {
    writer->WriteObjectKeyValue("index", index);
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("master", master);
    writer->WriteObjectKeyValue("optype", optype);
    writer->WriteObjectKeyValue("parents", parents);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("outputs", outputs);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("weights", weights);
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "index") {
        reader->Read(&index);
        bitmask |= 1;
      } else if (key == "name") {
        reader->Read(&name);
        bitmask |= 2;
      } else if (key == "master") {
        reader->Read(&master);
      } else if (key == "optype") {
        reader->Read(&optype);
        bitmask |= 4;
      } else if (key == "parents") {
        reader->Read(&parents);
      } else if (key == "inputs") {
        reader->Read(&inputs);
      } else if (key == "outputs") {
        reader->Read(&outputs);
        bitmask |= 4;
      } else if (key == "attrs") {
        reader->Read(&attrs);
      } else if (key == "weights") {
        reader->Read(&weights);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "index, name, optype and outputs should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for MSCGraph.
 *  MSCGraph is core of MSC.
 *  MSCGraph contains MSCJoints as nodes and MSCTensors as edges.
 */
struct JsonMSCGarph {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> nodes;

  const std::string Export() const {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    writer.BeginObject();
    Save(&writer);
    writer.EndObject();
    return os.str();
  }

  void Save(dmlc::JSONWriter* writer) const {
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("outputs", outputs);
    writer->WriteObjectKeyValue("nodes", nodes);
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "name") {
        reader->Read(&name);
        bitmask |= 1;
      } else if (key == "inputs") {
        reader->Read(&inputs);
        bitmask |= 2;
      } else if (key == "outputs") {
        reader->Read(&outputs);
        bitmask |= 4;
      } else if (key == "nodes") {
        reader->Read(&nodes);
        bitmask |= 8;
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "name, inputs, outputs and nodes should be given";
  }
};

/*!
 * \brief Tensor in MSCGraph.
 */
class MSCTensorNode : public Object {
 public:
  /*! \brief The name of tensor. */
  String name;
  /*! \brief The alias of tensor, can be changed. */
  mutable String alias;
  /*! \brief The data type of tensor. */
  DataType dtype;
  /*! \brief The layout of tensor. */
  tvm::tir::Layout layout;
  /*! \brief The shape of tensor. */
  Array<Integer> shape;
  /*! \brief Export tensor to json. */
  const String ToJson() const;
  /*! \brief Load tensor from json. */
  void FromJson(const std::string& json_str);
  /*! \brief Get the ndim of tensor. */
  size_t Ndim() const;
  /*! \brief Get dim at given index. */
  const Integer DimAt(int index) const;
  /*! \brief Get dim at given axis. */
  const Integer DimAt(const String& axis) const;
  /*! \brief Get size of the tensor. */
  const Integer GetSize() const;
  /*! \brief Get name of the dtype. */
  const String DtypeName() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("alias", &alias);
    v->Visit("dtype", &dtype);
    v->Visit("layout", &layout);
    v->Visit("shape", &shape);
  }

  bool SEqualReduce(const MSCTensorNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(dtype, other->dtype) && equal(shape, other->shape) &&
           equal(layout, other->layout);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(dtype);
    hash_reduce(shape);
    hash_reduce(layout);
  }

  static constexpr const char* _type_key = "msc.core.MSCTensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCTensorNode, Object);
};

/*!
 * \brief Managed reference to MSCTensorNode.
 * \sa MSCTensorNode
 */
class MSCTensor : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the tensor.
   * \param dtype The data type the tensor.
   * \param layout The layout of the tensor.
   * \param shape The shape of the tensor.
   * \param alias The alias of the tensor.
   */
  TVM_DLL MSCTensor(const String& name, const DataType& dtype, const String& layout,
                    const Array<Integer>& shape, const String& alias = "");

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the tensor.
   */
  TVM_DLL MSCTensor(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCTensor, ObjectRef, MSCTensorNode);
};

/*!
 * \brief Basic node in MSCGraph and WeightGraph.
 */
class BaseJoint;
class BaseJointNode : public Object {
 public:
  /*! \brief The index of node, can be changed. */
  mutable int index;
  /*! \brief The name of node. */
  String name;
  /*! \brief The master of node, can be changed. */
  String master;
  /*! \brief The op type of node. */
  String optype;
  /*! \brief The attributes of node. */
  Map<String, String> attrs;
  /*! \brief The parents of node. */
  Array<ObjectRef> parents;
  /*! \brief The children of node. */
  mutable Array<ObjectRef> children;
  /*! \brief Add child to the node. */
  size_t AddChild(const BaseJoint& child) const;
  /*! \brief Get the attribute by type. */
  bool GetAttr(const String& key, std::string* val) const;
  bool GetAttr(const String& key, int* val) const;
  bool GetAttr(const String& key, int64_t* val) const;
  bool GetAttr(const String& key, float* val) const;
  bool GetAttr(const String& key, bool* val) const;
  bool GetAttr(const String& key, std::vector<int>* val) const;
  bool GetAttr(const String& key, std::vector<int64_t>* val) const;
  bool GetAttr(const String& key, std::vector<float>* val) const;

  static constexpr const char* _type_key = "msc.core.BaseJoint";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseJointNode, Object);
};

/*!
 * \brief Managed reference to BaseJointNode.
 * \sa BaseJointNode
 */
class BaseJoint : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseJoint, ObjectRef, BaseJointNode);
};

/*!
 * \brief Node in MSCGraph.
 */
class MSCJoint;
class MSCJointNode : public BaseJointNode {
 public:
  /*! \brief The scope of node. */
  Array<String> scope;
  /*! \brief The inputs of node, can be changed. */
  Array<Array<Integer>> inputs;
  /*! \brief The outputs of node. */
  Array<MSCTensor> outputs;
  /*! \brief The weights of node. */
  Map<String, MSCTensor> weights;
  /*! \brief Export node to json. */
  const String ToJson() const;
  /*! \brief Load node from json. */
  void FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes);
  /*! \brief Get input from the node. */
  const MSCTensor InputAt(int index) const;
  /*! \brief Get output from the node. */
  const MSCTensor OutputAt(int index) const;
  /*! \brief Get weight from the node. */
  const MSCTensor WeightAt(const String& wtype) const;
  /*! \brief Get Producer from the node. */
  const MSCJoint ProducerOf(int index) const;
  const MSCJoint ProducerOf(const String& input_name) const;
  const MSCJoint ProducerOf(const MSCTensor& input) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("index", &index);
    v->Visit("name", &name);
    v->Visit("master", &master);
    v->Visit("optype", &optype);
    v->Visit("attrs", &attrs);
    v->Visit("parents", &parents);
    v->Visit("childern", &children);
    v->Visit("scope", &scope);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("weights", &weights);
  }

  bool SEqualReduce(const MSCJointNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(master, other->master) &&
           equal(optype, other->optype) && equal(attrs, other->attrs) &&
           equal(parents, other->parents) && equal(children, other->children) &&
           equal(scope, other->scope) && equal(inputs, other->inputs) &&
           equal(outputs, other->outputs) && equal(weights, other->weights);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(master);
    hash_reduce(optype);
    hash_reduce(attrs);
    hash_reduce(parents);
    hash_reduce(children);
    hash_reduce(scope);
    hash_reduce(inputs);
    hash_reduce(outputs);
    hash_reduce(weights);
  }

  static constexpr const char* _type_key = "msc.core.MSCJoint";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCJointNode, BaseJointNode);
};

/*!
 * \brief Managed reference to MSCJointNode.
 * \sa MSCJointNode
 */
class MSCJoint : public BaseJoint {
 public:
  /*!
   * \brief The constructor.
   * \param index The index of the node.
   * \param name The name of the node.
   * \param master The master of the node.
   * \param optype The op type the node.
   * \param attrs The attributes of the node.
   * \param inputs The inputs of the node.
   * \param outputs The outputs of the node.
   * \param weights The weights of the node.
   */
  TVM_DLL MSCJoint(int index, const String& name, const String& master, const String& optype,
                   const Map<String, String>& attrs, const Array<String>& scope,
                   const std::vector<std::pair<BaseJoint, size_t>>& inputs,
                   const Array<MSCTensor>& outputs, const Map<String, MSCTensor>& weights);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the node.
   */
  TVM_DLL MSCJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCJoint, BaseJoint, MSCJointNode);
};

/*!
 * \brief Node in WeightGraph.
 */
class WeightJoint;
class WeightJointNode : public BaseJointNode {
 public:
  /*! \brief The weight reference of weight node. */
  String wtype;
  /*! \brief The weight of weight node. */
  MSCTensor weight;
  /*! \brief The friends of weight node. */
  Array<BaseJoint> friends;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("index", &index);
    v->Visit("name", &name);
    v->Visit("master", &master);
    v->Visit("optype", &optype);
    v->Visit("attrs", &attrs);
    v->Visit("parents", &parents);
    v->Visit("children", &children);
    v->Visit("wtype", &wtype);
    v->Visit("weight", &weight);
    v->Visit("friends", &friends);
  }

  bool SEqualReduce(const WeightJointNode* other, SEqualReducer equal) const {
    return equal(name, other->name) &&
           equal(master, other->master) & equal(optype, other->optype) &&
           equal(attrs, other->attrs) && equal(parents, other->parents) &&
           equal(children, other->children) && equal(wtype, other->wtype) &&
           equal(weight, other->weight) && equal(friends, other->friends);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(master);
    hash_reduce(optype);
    hash_reduce(attrs);
    hash_reduce(parents);
    hash_reduce(children);
    hash_reduce(wtype);
    hash_reduce(weight);
    hash_reduce(friends);
  }

  static constexpr const char* _type_key = "msc.core.WeightJoint";
  TVM_DECLARE_FINAL_OBJECT_INFO(WeightJointNode, BaseJointNode);
};

/*!
 * \brief Managed reference to WeightJointNode.
 * \sa WeightJointNode
 */
class WeightJoint : public BaseJoint {
 public:
  /*!
   * \brief The constructor.
   * \param index The index of the node.
   * \param name The name of the node.
   * \param master The master of the node.
   * \param optype The optype of the node.
   * \param wtype The weight type of the node.
   * \param attrs The attributes of the node.
   * \param weight The weight tensor of the node.
   * \param parents The parents of the node.
   * \param friends The friends of the node.
   */
  TVM_DLL WeightJoint(int index, const String& name, const String& master, const String& optype,
                      const String& wtype, const Map<String, String>& attrs,
                      const MSCTensor& weight, const Array<BaseJoint> parents,
                      const Array<BaseJoint>& friends);

  TVM_DEFINE_OBJECT_REF_METHODS(WeightJoint, BaseJoint, WeightJointNode);
};

/*!
 * \brief Basic graph class (MSCGraph and WeightGraph).
 */
class BaseGraphNode : public Object {
 public:
  /*! \brief The name of graph. */
  String name;
  /*! \brief The node names in graph, can be changed. */
  Array<String> node_names;
  /*! \brief The nodes in graph, can be changed. */
  Map<String, BaseJoint> nodes;

  static constexpr const char* _type_key = "msc.core.BaseGraph";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseGraphNode, Object);
};

/*!
 * \brief Managed reference to BaseGraphNode.
 * \sa BaseGraphNode
 */
class BaseGraph : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseGraph, ObjectRef, BaseGraphNode);
};

/*!
 * \brief MSCGraph.
 */
class MSCGraph;
class MSCGraphNode : public BaseGraphNode {
 public:
  /*! \brief The input names of graph. */
  Array<String> input_names;
  /*! \brief The output names of graph. */
  Array<String> output_names;
  /*! \brief The tensor alias in graph, get by AnalysisGraph. */
  Map<String, String> tensor_alias;
  /*! \brief The weights in graph, get by AnalysisGraph. */
  Map<String, Array<String>> weight_holders;
  /*! \brief Export graph to json. */
  const String ToJson() const;
  /*! \brief Load graph from json. */
  void FromJson(const std::string& json_str);
  /*! \brief Export graph to prototxt. */
  const String ToPrototxt() const;
  /*! \brief Find node in graph. */
  const MSCJoint FindNode(const String& name) const;
  /*! \brief Get input from the graph. */
  const MSCTensor InputAt(int index) const;
  /*! \brief Get output from the graph. */
  const MSCTensor OutputAt(int index) const;
  /*! \brief Find tensor from the graph. */
  const MSCTensor FindTensor(const String& name) const;
  /*! \brief Find producer of tensor from the graph. */
  const MSCJoint FindProducer(const String& name) const;
  /*! \brief Find producer of tensor from the graph. */
  const MSCJoint FindProducer(const MSCTensor& tensor) const;
  /*! \brief Find producer and output index of tensor from the graph. */
  const std::pair<MSCJoint, size_t> FindProducerAndIdx(const String& name) const;
  /*! \brief Find producer and output index of tensor from the graph. */
  const std::pair<MSCJoint, size_t> FindProducerAndIdx(const MSCTensor& tensor) const;
  /*! \brief Find consumers of tensor from the graph. */
  const Array<MSCJoint> FindConsumers(const String& name) const;
  /*! \brief Find consumers of tensor from the graph. */
  const Array<MSCJoint> FindConsumers(const MSCTensor& tensor) const;
  /*! \brief Find consumers and input indices of tensor from the graph. */
  const std::vector<std::pair<MSCJoint, size_t>> FindConsumersAndIndices(const String& name) const;
  /*! \brief Find consumers and input indices of tensor from the graph. */
  const std::vector<std::pair<MSCJoint, size_t>> FindConsumersAndIndices(
      const MSCTensor& tensor) const;
  /*! \brief Analysis the graph and fill info. */
  void AnalysisGraph();

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("nodes", &nodes);
    v->Visit("node_names", &node_names);
    v->Visit("input_names", &input_names);
    v->Visit("output_names", &output_names);
    v->Visit("weight_holders", &weight_holders);
  }

  bool SEqualReduce(const MSCGraphNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(nodes, other->nodes) &&
           equal(node_names, other->node_names) && equal(input_names, other->input_names) &&
           equal(output_names, other->output_names) && equal(weight_holders, other->weight_holders);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(nodes);
    hash_reduce(node_names);
    hash_reduce(input_names);
    hash_reduce(output_names);
    hash_reduce(weight_holders);
  }

  static constexpr const char* _type_key = "msc.core.MSCGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCGraphNode, BaseGraphNode);
};

/*!
 * \brief Managed reference to MSCGraphNode.
 * \sa MSCGraphNode
 */
class MSCGraph : public BaseGraph {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the node.
   * \param node_names The node names in the graph
   * \param nodes The nodes in the graph.
   * \param input_names The input names of the graph.
   * \param output_names The output names of the graph.
   * \param weight_holders The weights info of the graph.
   */
  TVM_DLL MSCGraph(const String& name, const Array<MSCJoint>& nodes,
                   const Array<String>& input_names, const Array<String>& output_names);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the graph.
   */
  TVM_DLL MSCGraph(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCGraph, BaseGraph, MSCGraphNode);
};

/*!
 * \brief WeightGraph.
 */
class WeightGraphNode : public BaseGraphNode {
 public:
  /*! \brief Export graph to prototxt. */
  const String ToPrototxt() const;
  /*! \brief Find node in graph. */
  const WeightJoint FindNode(const String& name) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("nodes", &nodes);
    v->Visit("node_names", &node_names);
  }

  bool SEqualReduce(const WeightGraphNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(nodes, other->nodes) &&
           equal(node_names, other->node_names);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(nodes);
    hash_reduce(node_names);
  }

  static constexpr const char* _type_key = "msc.core.WeightGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(WeightGraphNode, BaseGraphNode);
};

/*!
 * \brief Managed reference to WeightGraphNode.
 * \sa WeightGraphNode
 */
class WeightGraph : public BaseGraph {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the node.
   * \param node_names The node names in the graph
   * \param nodes The nodes in the graph.
   */
  TVM_DLL WeightGraph(const String& name, const Array<WeightJoint>& node_names);

  TVM_DEFINE_OBJECT_REF_METHODS(WeightGraph, BaseGraph, WeightGraphNode);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_IR_MSC_GRAPH_H_