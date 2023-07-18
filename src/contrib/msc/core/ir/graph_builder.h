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
 * \file src/contrib/msc/core/ir/graph_builder.h
 * \brief Builder of MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_
#define TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_

#include <dmlc/json.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/tir/data_layout.h>

#include "../utils.h"
#include "graph.h"

namespace tvm {
namespace contrib {
namespace msc {

using Expr = tvm::RelayExpr;
using RelaxExprVisitor = tvm::relax::ExprVisitor;
using RelayExprVisitor = tvm::relay::ExprVisitor;
using namespace tvm::runtime;

/*!
 * \brief Config for building MSCGraph.
 *  Define the configuration for building MSCGraph
 */
struct MSCRBuildConfig {
  bool prune_graph{false};
  std::string sort_by;
  std::vector<std::string> input_aliass;
  std::vector<std::string> output_aliass;
  std::unordered_map<std::string, std::vector<std::string>> input_types;

  void LoadInputTypes(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      std::vector<std::string> types;
      reader->Read(&types);
      input_types[key] = types;
    }
  }

  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "prune_graph") {
        reader->Read(&prune_graph);
      } else if (key == "sort_by") {
        reader->Read(&sort_by);
      } else if (key == "input_aliass") {
        reader->Read(&input_aliass);
      } else if (key == "output_aliass") {
        reader->Read(&output_aliass);
      } else if (key == "input_types") {
        this->LoadInputTypes(reader);
      }
    }
  }
};

class AttrGetter : public AttrVisitor {
 public:
  /*!
   * \brief Get the attributes as Map<String, String>
   * \param keys the keys.
   * \param values the values.
   */
  explicit AttrGetter(Array<String>* keys, Array<String>* values) : keys_(keys), values_(values) {}

  void Visit(const char* key, double* value) final {
    keys_->push_back(key);
    values_->push_back(std::to_string(*value));
  }

  void Visit(const char* key, int64_t* value) final {
    keys_->push_back(key);
    values_->push_back(std::to_string(*value));
  }

  void Visit(const char* key, uint64_t* value) final {
    keys_->push_back(key);
    values_->push_back(std::to_string(*value));
  }

  void Visit(const char* key, int* value) final {
    keys_->push_back(key);
    values_->push_back(std::to_string(*value));
  }

  void Visit(const char* key, bool* value) final {
    keys_->push_back(key);
    values_->push_back(std::to_string(*value));
  }

  void Visit(const char* key, std::string* value) final {
    keys_->push_back(key);
    values_->push_back(*value);
  }

  void Visit(const char* key, DataType* value) final {
    keys_->push_back(key);
    values_->push_back(runtime::DLDataType2String(*value));
  }

  void Visit(const char* key, runtime::ObjectRef* value) final {
    keys_->push_back(key);
    values_->push_back(StringUtils::ToString(*value));
  }

  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "TypeError: void is not allowed in Attrs";
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "TypeError: NDArray is not allowed in Attrs";
  }

 private:
  Array<String>* keys_;
  Array<String>* values_;
};

class RelaxGraphBuilder : public RelaxExprVisitor {
 public:
  /*!
   * \brief The constructor of RelaxGraphBuilder
   * \param options the options for the builder.
   * \param indent the start indent.
   */
  explicit RelaxGraphBuilder(const String& name, const std::string& options = "")
      : RelaxExprVisitor() {
    name_ = name;
    if (options.size() > 0) {
      std::istringstream is(options);
      dmlc::JSONReader reader(&is);
      reader.Read(&config_);
    }
  }

  MSCGraph Build(const relax::Function& func);

  MSCJoint AddNode(const Expr& expr, const Optional<Expr>& binding_var = NullOpt,
                   const String& name = "");

  void VisitBindingBlock(const relax::BindingBlock& block) final;

  void VisitExpr_(const relax::ConstantNode* op) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::CallNode* call_node) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::TupleNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding,
                     const relax::TupleGetItemNode* val) final;

  void VisitBinding_(const relax::VarBindingNode* binding, const relax::DataflowVarNode* val) final;

 private:
  String name_;
  String scope_name_;
  MSCRBuildConfig config_;
  Array<MSCJoint> nodes_;
  Map<String, MSCTensor> weights_;
  Map<Expr, Array<String>> expr_tensor_map_;
  std::unordered_map<String, std::pair<BaseJoint, size_t>> tensor_input_map_;
};

class RelaxWeightsExtractor : public RelaxExprVisitor {
 public:
  /*!
   * \brief Visit the constant and save weights
   */
  Map<String, NDArray> GetWeights(const relax::Function& func);

  void VisitExpr_(const relax::ConstantNode* op) final;

 private:
  Map<String, NDArray> weights_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_IR_GRAPH_BUILDER_H_
