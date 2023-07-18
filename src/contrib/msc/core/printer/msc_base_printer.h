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
 * \file src/contrib/msc/core/printer/msc_base_printer.h
 * \brief Base Printer for all MSC printers.
 */
#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_MSC_BASE_PRINTER_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_MSC_BASE_PRINTER_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include "../../../../../src/support/str_escape.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

struct MSCPrinterConfig {
  std::string space{"  "};
  std::string separator{", "};
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "space") {
        reader->Read(&space);
      } else if (key == "separator") {
        reader->Read(&separator);
      } else {
        LOG(FATAL) << "Do not support config " << key << " in printer";
      }
    }
  }
};

/*!
 * \brief BasePrinterConfig is base for config class in MSC
 * \sa Doc
 */

/*!
 * \brief MSCBasePrinter is responsible for printing Doc tree into text format
 * \sa Doc
 */
class MSCBasePrinter {
 public:
  /*!
   * \brief The constructor of DocPrinter
   * \param options the options for the printer.
   * \param indent the start indent.
   */
  explicit MSCBasePrinter(const std::string& options = "", size_t indent = 0) {
    indent_ = indent;
    if (options.size() > 0) {
      std::istringstream is(options);
      dmlc::JSONReader reader(&is);
      reader.Read(&config_);
    }
  }

  virtual ~MSCBasePrinter() = default;

  /*!
   * \brief Append a doc into the final content
   * \sa GetString
   */
  void Append(const Doc& doc) { PrintDoc(doc); }

  /*!
   * \brief Get the printed string of all Doc appended
   * \sa Append
   */
  String GetString() const { return output_.str(); }

 protected:
  /*!
   * \brief Get the printed string
   * \sa PrintTypedDoc
   */
  void PrintDoc(const Doc& doc) {
    if (auto doc_node = doc.as<LiteralDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<IdDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<AttrAccessDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<IndexDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<CallDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<ListDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<TupleDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<DictDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<SliceDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<AssignDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<IfDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<WhileDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<ForDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<ScopeDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<AssertDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<ReturnDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<FunctionDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<ClassDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else if (auto doc_node = doc.as<CommentDoc>()) {
      PrintTypedDoc(doc_node.value());
    } else {
      LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    }
  }

  template <typename DocType>
  void PrintJoinedDocs(const Array<DocType>& docs) {
    for (size_t i = 0; i < docs.size(); i++) {
      PrintDoc(docs[i]);
      output_ << (i == docs.size() - 1 ? "" : config_.separator);
    }
  }

  /*!
   * \brief Virtual method to print a LiteralDoc
   */
  virtual void PrintTypedDoc(const LiteralDoc& doc) {
    const ObjectRef& value = doc->value;
    if (!value.defined()) {
      output_ << "\"\"";
    } else if (const auto* int_imm = value.as<IntImmNode>()) {
      output_ << int_imm->value;
    } else if (const auto* float_imm = value.as<FloatImmNode>()) {
      // TODO(yelite): Make float number printing roundtrippable
      output_.precision(17);
      if (std::isinf(float_imm->value) || std::isnan(float_imm->value)) {
        output_ << '"' << float_imm->value << '"';
      } else {
        output_ << float_imm->value;
      }
    } else if (const auto* string_obj = value.as<StringObj>()) {
      output_ << "\"" << tvm::support::StrEscape(string_obj->data, string_obj->size) << "\"";
    } else {
      LOG(FATAL) << "TypeError: Unsupported literal value type: " << value->GetTypeKey();
    }
  }

  /*!
   * \brief Virtual method to print an IdDoc
   */
  virtual void PrintTypedDoc(const IdDoc& doc) { output_ << doc->name; }

  /*!
   * \brief Virtual method to print an IndexDoc
   */
  virtual void PrintTypedDoc(const IndexDoc& doc) {
    PrintDoc(doc->value);
    if (doc->indices.size() == 0) {
      output_ << "[()]";
    } else {
      for (size_t i = 0; i < doc->indices.size(); i++) {
        if (i == 0) {
          output_ << "[";
        }
        PrintDoc(doc);
        output_ << (i == doc->indices.size() - 1 ? "]" : config_.separator);
      }
    }
  }

  /*!
   * \brief Virtual method to print a CallDoc
   */
  virtual void PrintTypedDoc(const CallDoc& doc) {
    PrintDoc(doc->callee);
    output_ << "(";
    PrintJoinedDocs(doc->args);
    ICHECK_EQ(doc->kwargs_keys.size(), doc->kwargs_values.size())
        << "CallDoc should have equal number of elements in kwargs_keys and kwargs_values.";
    if (doc->args.size() > 0 && doc->kwargs_keys.size() > 0) {
      output_ << config_.separator;
    }
    for (size_t i = 0; i < doc->kwargs_keys.size(); i++) {
      output_ << doc->kwargs_keys[i] << "=";
      PrintDoc(doc->kwargs_values[i]);
      output_ << (i == doc->kwargs_keys.size() - 1 ? "" : config_.separator);
    }
    output_ << ")";
  }

  /*!
   * \brief Virtual method to print a ListDoc
   */
  virtual void PrintTypedDoc(const ListDoc& doc) {
    output_ << "[";
    PrintJoinedDocs(doc->elements);
    output_ << "]";
  }

  /*!
   * \brief Virtual method to print a TupleDoc
   */
  virtual void PrintTypedDoc(const TupleDoc& doc) {
    output_ << "(";
    if (doc->elements.size() == 1) {
      PrintDoc(doc->elements[0]);
      output_ << ",";
    } else {
      PrintJoinedDocs(doc->elements);
    }
    output_ << ")";
  }

  /*!
   * \brief Virtual method to print a ReturnDoc
   */
  virtual void PrintTypedDoc(const ReturnDoc& doc) {
    output_ << "return ";
    PrintDoc(doc->value);
  }

  /*!
   * \brief Virtual method to print an AttrAccessDoc
   */
  virtual void PrintTypedDoc(const AttrAccessDoc& doc) {
    LOG(FATAL) << "AttrAccessDoc is not implemented";
  }

  /*!
   * \brief Virtual method to print a DictDoc
   */
  virtual void PrintTypedDoc(const DictDoc& doc) { LOG(FATAL) << "DictDoc is not implemented"; }

  /*!
   * \brief Virtual method to print a SliceDoc
   */
  virtual void PrintTypedDoc(const SliceDoc& doc) { LOG(FATAL) << "SliceDoc is not implemented"; }

  /*!
   * \brief Virtual method to print an AssignDoc
   */
  virtual void PrintTypedDoc(const AssignDoc& doc) { LOG(FATAL) << "AssignDoc is not implemented"; }

  /*!
   * \brief Virtual method to print an IfDoc
   */
  virtual void PrintTypedDoc(const IfDoc& doc) { LOG(FATAL) << "IfDoc is not implemented"; }

  /*!
   * \brief Virtual method to print a WhileDoc
   */
  virtual void PrintTypedDoc(const WhileDoc& doc) { LOG(FATAL) << "WhileDoc is not implemented"; }

  /*!
   * \brief Virtual method to print a ForDoc
   */
  virtual void PrintTypedDoc(const ForDoc& doc) { LOG(FATAL) << "ForDoc is not implemented"; }

  /*!
   * \brief Virtual method to print a ScopeDoc
   */
  virtual void PrintTypedDoc(const ScopeDoc& doc) { LOG(FATAL) << "ScopeDoc is not implemented"; }

  /*!
   * \brief Virtual method to print an AssertDoc
   */
  virtual void PrintTypedDoc(const AssertDoc& doc) { LOG(FATAL) << "AssertDoc is not implemented"; }

  /*!
   * \brief Virtual method to print a FunctionDoc
   */
  virtual void PrintTypedDoc(const FunctionDoc& doc) {
    LOG(FATAL) << "FunctionDoc is not implemented";
  }

  /*!
   * \brief Virtual method to print a ClassDoc
   */
  virtual void PrintTypedDoc(const ClassDoc& doc) { LOG(FATAL) << "ClassDoc is not implemented"; }

  /*!
   * \brief Virtual method to print a CommentDoc
   */
  virtual void PrintTypedDoc(const CommentDoc& doc) {
    LOG(FATAL) << "CommentDoc is not implemented";
  }

  /*!
   * \brief Start the line into the output stream.
   * \sa output_
   */
  void StartLine() {
    for (size_t i = 0; i < indent_; i++) {
      output_ << config_.space;
    }
  }

  /*!
   * \brief Add a new line into the output stream.
   * \sa output_
   */
  void NewLine() { output_ << "\n"; }

  /*!
   * \brief Increase the indent level
   */
  void IncreaseIndent() { indent_ += 1; }

  /*!
   * \brief Decrease the indent level
   */
  void DecreaseIndent() {
    if (indent_ >= 1) {
      indent_ -= 1;
    }
  }

  /*! \brief The output stream of printer*/
  std::ostringstream output_;
  /*! \brief the config for printer */
  MSCPrinterConfig config_;

 private:
  /*! \brief the current level of indent */
  size_t indent_ = 0;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_MSC_BASE_PRINTER_H_
