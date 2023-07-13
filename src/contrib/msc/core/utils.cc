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
 * \file utils.cc
 */
#include "utils.h"

namespace tvm {
namespace contrib {
namespace msc {

Array<String> StringUtils::Split(const String& src_string, const String& sep) {
  Array<String> sub_strings;
  if (src_string.size() == 0) {
    return sub_strings;
  }
  std::string src_cstring = src_string;
  const std::string& csep = sep;
  int pos = src_cstring.find(csep);
  while (pos > 0) {
    sub_strings.push_back(src_cstring.substr(0, pos));
    src_cstring = src_cstring.substr(pos + csep.size() + 1);
    pos = src_cstring.find(csep);
  }
  return sub_strings;
}

std::tuple<String, String> StringUtils::SplitOnce(const String& src_string, const String& sep,
                                                  bool from_left) {
  if (src_string.size() == 0) {
    return std::make_tuple(String(), String());
  }
  std::string src_cstring = src_string;
  const std::string& csep = sep;
  int pos = from_left ? src_cstring.find(csep) : src_cstring.rfind(csep);
  if (pos >= 0) {
    return std::make_tuple(src_cstring.substr(0, pos), src_cstring.substr(pos + csep.size()));
  }
  return std::make_tuple(String(), String());
}

Array<String> StringUtils::GetClosures(const String& src_string, const String& left,
                                       const String& right) {
  Array<String> tokens;
  if (src_string.size() == 0) {
    return tokens;
  }
  String token = "start";
  String left_str = src_string;
  while (token.size() > 0) {
    std::tie(token, left_str) = StringUtils::SplitOnce(left_str, left);
    if (left_str.size() > 0) {
      std::tie(token, left_str) = StringUtils::SplitOnce(left_str, right);
    } else {
      token = "";
    }
    if (token.size() > 0) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

String StringUtils::GetClosureOnce(const String& src_string, const String& left,
                                   const String& right, bool from_left) {
  if (src_string.size() == 0) {
    return "";
  }
  String val = std::get<1>(SplitOnce(src_string, left, from_left));
  if (val.size() > 0) {
    val = std::get<0>(StringUtils::SplitOnce(val, right, from_left));
  }
  return val;
}

Span SpanUtils::SetAttr(const Span& span, const String& key, const String& value) {
  if (value.size() == 0) {
    return span;
  }
  String new_source;
  Array<String> tokens{"<" + key + ">", "</" + key + ">"};
  if (span.defined() && span->source_name.defined()) {
    const String& source_str = span->source_name->name;
    String left = std::get<0>(StringUtils::SplitOnce(source_str, tokens[0]));
    String right = std::get<1>(StringUtils::SplitOnce(source_str, tokens[1]));
    if (left.size() > 0 && right.size() > 0) {
      new_source = left + tokens[0] + value + tokens[1] + right;
    } else {
      new_source = source_str + tokens[0] + value + tokens[1];
    }
  } else {
    new_source = tokens[0] + value + tokens[1];
  }
  if (span.defined()) {
    return Span(SourceName::Get(new_source), span->line, span->end_line, span->column,
                span->end_column);
  }
  return Span(SourceName::Get(new_source), 0, 0, 0, 0);
}

String SpanUtils::GetAttr(const Span& span, const String& key) {
  if (span.defined() && span->source_name.defined()) {
    Array<String> tokens{"<" + key + ">", "</" + key + ">"};
    return StringUtils::GetClosureOnce(span->source_name->name, tokens[0], tokens[1]);
  }
  return "";
}

Map<String, String> SpanUtils::GetAttrs(const Span& span) {
  Map<String, String> attrs;
  for (const auto& key : StringUtils::GetClosures(span->source_name->name, "</", ">")) {
    attrs.Set(key, GetAttr(span, key));
  }
  return attrs;
}

Array<String> ExprUtils::GetInputTypes(const Call& call) {
  Array<String> input_types;
  const String& op_name = Downcast<Op>(call->op)->name;
  if (op_name == "relax.nn.conv1d" || op_name == "relax.nn.conv2d" ||
      op_name == "relax.nn.conv3d" || op_name == "relax.nn.dense") {
    input_types.push_back("input");
    input_types.push_back("weight");
  } else if (op_name == "relax.nn.batch_norm") {
    input_types.push_back("input");
    input_types.push_back("gamma");
    input_types.push_back("beta");
    input_types.push_back("mean");
    input_types.push_back("var");
  } else if (op_name == "relax.nn.layer_norm") {
    input_types.push_back("input");
    input_types.push_back("gamma");
    input_types.push_back("beta");
  } else {
    for (size_t i = 0; i < call->args.size(); i++) {
      input_types.push_back("input");
    }
  }
  ICHECK_EQ(input_types.size(), call->args.size()) << "Input types and args size mismatch";
  return input_types;
}

TVM_REGISTER_GLOBAL("msc.core.SpanGetAttr").set_body_typed(SpanUtils::GetAttr);

TVM_REGISTER_GLOBAL("msc.core.SpanGetAttrs").set_body_typed(SpanUtils::GetAttrs);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm