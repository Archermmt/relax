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
 * \file utils.h
 * \brief Common utilities for msc.
 */
#ifndef TVM_CONTRIB_MSC_CORE_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_UTILS_H_

#include <tvm/ir/source_map.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace contrib {
namespace msc {

using Expr = tvm::RelayExpr;
using RelaxCall = tvm::relax::Call;
using RelayCall = tvm::relay::Call;

class CommonUtils {
 public:
  /*!
   * \brief Check if the index is in range.
   * \return The valid index.
   */
  TVM_DLL static size_t GetIndex(int index, size_t max_size);
};

/*!
 * \brief Utils for String.
 */
class StringUtils {
 public:
  /*!
   * \brief Split the String into sub Strings.
   * \return The SubStrings.
   */
  TVM_DLL static Array<String> Split(const String& src_string, const String& sep);

  /*!
   * \brief Join the SubStrings into String.
   * \return The String.
   */
  TVM_DLL static String Join(const Array<String>& sub_strings, const String& joint);

  /*!
   * \brief Replace the substring ole to new in String.
   * \return The replaced String.
   */
  TVM_DLL static String Replace(const String& src_string, const String& old_str,
                                const String& new_str);

  /*!
   * \brief Split the String into two sub Strings, only split by the frist seq.
   * \return The SubStrings.
   */
  TVM_DLL static std::tuple<String, String> SplitOnce(const String& src_string, const String& sep,
                                                      bool from_left = true);

  /*!
   * \brief Get the tokens between left and right.
   * \return The Tokens.
   */
  TVM_DLL static Array<String> GetClosures(const String& src_string, const String& left,
                                           const String& right);

  /*!
   * \brief Get the first token between left and right.
   * \return The Token.
   */
  TVM_DLL static String GetClosureOnce(const String& src_string, const String& left,
                                       const String& right, bool from_left = true);

  /*!
   * \brief Change Object to String.
   * \return The String.
   */
  TVM_DLL static String ToString(const runtime::ObjectRef& obj);
};

/*!
 * \brief Utils for Span.
 */
class SpanUtils {
 public:
  /*!
   * \brief Set <key>value</key> to the Span.
   * \return The new Span.
   */
  TVM_DLL static Span SetAttr(const Span& span, const String& key, const String& value);

  /*!
   * \brief Get the value in <key>value</key> from the Span.
   * \return The value String.
   */
  TVM_DLL static String GetAttr(const Span& span, const String& key);

  /*!
   * \brief Get all the key:value in format <key>value</key> from the Span.
   * \return The Attrs Map.
   */
  TVM_DLL static Map<String, String> GetAttrs(const Span& span);
};

/*!
 * \brief Utils for Expr.
 */
class ExprUtils {
 public:
  /*!
   * \brief Get the input types of call.
   * \return The input types.
   */
  TVM_DLL static Array<String> GetInputTypes(const String& optype);

  /*!
   * \brief Get the input types of call.
   * \return The input types.
   */
  TVM_DLL static Array<String> GetInputTypes(const RelaxCall& call);

  /*!
   * \brief Get the input types of call.
   * \return The input types.
   */
  TVM_DLL static Array<String> GetInputTypes(const RelayCall& call);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_UTILS_H_
