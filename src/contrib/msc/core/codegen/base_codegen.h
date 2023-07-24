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
 * \file src/contrib/msc/core/codegen/base_codegen.h
 * \brief Basic CodeGen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_BASE_CODEGEN_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_BASE_CODEGEN_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include "../ir/graph.h"
#include "code_stack.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

#define CODEGEN_CONFIG_MEMBERS             \
  bool is_train{false};                    \
  bool need_prune{false};                  \
  bool need_quantize{false};               \
  bool need_collect{false};                \
  bool need_distill{false};                \
  bool need_process{false};                \
  bool need_test{true};                    \
  std::string prefix{"res_"};              \
  std::string baseline_folder{"baseline"}; \
  std::string version;

#define CODEGEN_CONFIG_PARSE                    \
  if (key == "is_train") {                      \
    reader->Read(&is_train);                    \
  } else if (key == "need_prune") {             \
    reader->Read(&need_prune);                  \
    need_process |= need_prune;                 \
  } else if (key == "need_quantize") {          \
    reader->Read(&need_quantize);               \
    need_process |= need_quantize;              \
  } else if (key == "need_collect") {           \
    reader->Read(&need_collect);                \
    need_process |= need_collect;               \
  } else if (key == "need_distill") {           \
    reader->Read(&need_distill);                \
    need_process |= need_distill;               \
  } else if (key == "need_test") {              \
    reader->Read(&need_test);                   \
  } else if (key == "version") {                \
    reader->Read(&version);                     \
  } else if (key == "prefix") {                 \
    reader->Read(&prefix);                      \
  } else if (key == "baseline_folder") {        \
    reader->Read(&baseline_folder);             \
  } else {                                      \
    LOG(FATAL) << "Do not support key " << key; \
  }

#define CODEGEN_METHODS                                                                            \
  const String GetSuffix(bool as_raw = false) {                                                    \
    const String& suffix = as_raw && config()->need_process ? "_raw" : "";                         \
    return suffix;                                                                                 \
  };                                                                                               \
  virtual const String IdxNode(const MSCJoint& node, bool as_raw = true) {                         \
    return CodeGenUtils::IdxNode(node, config()->prefix, GetSuffix(as_raw));                       \
  };                                                                                               \
  virtual const String IdxInput(const MSCJoint& node, int idx = 0, bool as_raw = false) {          \
    return CodeGenUtils::IdxInput(node, config()->prefix, idx, GetSuffix(as_raw));                 \
  };                                                                                               \
  virtual const String IdxOutput(const MSCJoint& node, int idx = 0, bool as_raw = false) {         \
    return CodeGenUtils::IdxOutput(node, config()->prefix, idx, GetSuffix(as_raw));                \
  };                                                                                               \
  virtual const String IdxWeight(const MSCJoint& node, const String& wtype, bool as_raw = false) { \
    return CodeGenUtils::IdxWeight(node, wtype, GetSuffix(as_raw));                                \
  };                                                                                               \
  virtual const String DType(const DataType& dtype) { return runtime::DLDataType2String(dtype); }  \
  virtual const String Comment(const MSCJoint& node) {                                             \
    return CodeGenUtils::CommentNode(node, config()->prefix);                                      \
  }

/*!
 * \brief CodeGen for MSCJoint op
 */

template <typename ConfigType>
class BaseOpCodeGen {
 public:
  /*!
   * \brief The constructor of BaseOpCodeGen
   * \param func_name the function name for the node.
   * \param config the config json for the node.
   */
  explicit BaseOpCodeGen(const String& func_name, const std::shared_ptr<ConfigType> config)
      : func_name_(func_name), config_(config) {}

  /*! \brief Config the BaseOpCodeGen*/
  void Config(const MSCJoint& node) { node_ = node; }

  /*! \brief Convert node to docs*/
  virtual const Array<Doc> CodeGen() = 0;

  CODEGEN_METHODS;

  /*! \brief Get return describe for default node*/
  virtual const String IdxNode(bool as_raw = true) { return IdxNode(node_, as_raw); };

  /*! \brief Get describe for default node input*/
  virtual const String IdxInput(int idx = 0, bool as_raw = false) {
    return IdxInput(node_, idx, as_raw);
  };

  /*! \brief Get describe for default node output*/
  virtual const String IdxOutput(int idx = 0, bool as_raw = false) {
    return IdxOutput(node_, idx, as_raw);
  };

  /*! \brief Get describe for default node weight*/
  virtual const String IdxWeight(const String& wtype, bool as_raw = false) {
    return IdxWeight(node_, wtype, as_raw);
  }

  /*! \brief Get func_name for the default node*/
  const String func_name() { return func_name_; }

  /*! \brief Get the config*/
  const std::shared_ptr<ConfigType> config() { return config_; }

  /*! \brief Get the default node*/
  const MSCJoint node() { return node_; }

 private:
  String func_name_;
  std::shared_ptr<ConfigType> config_;
  MSCJoint node_;
};

/*!
 * \brief CodeGen for MSCGraph
 */
template <typename ConfigType>
class BaseGraphCodeGen {
 public:
  /*!
   * \brief The constructor of BaseGraphCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit BaseGraphCodeGen(const MSCGraph& graph, const std::string& config = "") {
    graph_ = graph;
    config_.reset(new ConfigType());
    if (config.size() > 0) {
      std::istringstream is(config);
      dmlc::JSONReader reader(&is);
      reader.Read(config_.get());
    }
  }

  /*! \brief Stack the docs for the script*/
  virtual void CodeGen() = 0;

  /*! \brief Get sources*/
  virtual const Map<String, String> GetSources(const std::string& print_options = "") const = 0;

  CODEGEN_METHODS;

 protected:
  /*! \brief Get the docs from last block*/
  const Array<Doc> GetDocs() const { return stack_.GetDocs(); }

  /*! \brief Get the docs for the op*/
  virtual const Array<Doc> GetOpCodes(const MSCJoint& node,
                                      const std::shared_ptr<ConfigType>& config) = 0;

  /*! \brief Get the graph*/
  const MSCGraph graph() const { return graph_; }

  /*! \brief Get the config*/
  const std::shared_ptr<ConfigType> config() { return config_; }

  /*! \brief The stack_ of codes*/
  CodeStack stack_;

 private:
  MSCGraph graph_;
  std::shared_ptr<ConfigType> config_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_BASE_CODEGEN_H_
