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
 * \file src/contrib/msc/framework/tvm/relax_opcode.cc
 */
#include "relax_opcode.h"

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> RelaxOpCodeGen::GetDocs() {
  auto stack = RelaxOpCodeStack(this);
  CodeGenBuild(stack);
  bool emit_var = true;
  if (node()->optype == "input" || node()->optype == "constant" || node()->optype == "shape") {
    emit_var = false;
  }
  if (node()->optype == "tuple" && node()->children.size() == 0) {
    emit_var = false;
  }
  if (emit_var) {
    const auto& name = config()->explicit_name ? node()->name : "";
    BuilderEmit(stack, IdxNode(true), name);
  }
  return stack.GetDocs();
}

void RelaxOpCodeGen::BuilderEmit(RelaxOpCodeStack& stack, const String& ret, const String& name) {
  stack.call_start("block_builder.emit").call_arg(ret);
  if (name.size() > 0) {
    stack.call_str_arg(name, "name_hint");
  }
  stack.call_end(ret);
}

#define RELAX_OP_CODEGEN_METHODS(TypeName) \
 public:                                   \
  TypeName(const String& func_name) : RelaxOpCodeGen(func_name) {}

class RelaxAdaptivePool2dCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxAdaptivePool2dCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_list_arg<int>("output_size")
        .op_str_arg("layout")
        .op_str_arg("out_layout")
        .op_end();
  }
};

class RelaxAstypeCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxAstypeCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_str_arg("dtype").op_end();
  }
};

class RelaxAttentionCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxAttentionCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    for (size_t i = 0; i < 3; i++) {
      const String& axes_key = i == 0 ? "axes" : "axes_" + std::to_string(i);
      stack.op_start("relax.op.permute_dims")
          .op_input_arg(i)
          .op_list_arg<int>(axes_key, "axes")
          .op_end(IdxInput(i));
    }
    stack.op_start().op_inputs_arg(false).op_arg<float>("scale").op_str_arg("causal_mask").op_end();
  }
};

class RelaxAxisCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxAxisCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_arg<int>("axis").op_end();
  }
};

class RelaxAxesCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxAxesCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    const String& key = node()->HasAttr("axes") ? "axes" : "axis";
    stack.op_start().op_input_arg().op_list_arg<int>(key).op_end();
  }
};

class RelaxBatchNormCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxBatchNormCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_weight_arg("gamma")
        .op_weight_arg("beta")
        .op_weight_arg("mean")
        .op_weight_arg("var")
        .op_arg<int>("axis")
        .op_arg<float>("epsilon")
        .op_arg<bool>("center")
        .op_arg<bool>("scale")
        .op_arg<float>("momentum")
        .op_end();
  }
};

class RelaxGetItemCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxGetItemCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_arg<int>("index").call_end(IdxNode());
  }
};

class RelaxBroadcastToCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxBroadcastToCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_list_arg<int>("shape").op_end();
  }
};

class RelaxClipCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxClipCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_arg<float>("min").op_arg<float>("max").op_end();
  }
};

class RelaxConstantCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxConstantCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .call_str_arg(node()->name)
        .call_inplace_start("relax.TensorStructInfo")
        .call_list_arg(node()->OutputAt(0)->shape)
        .call_str_arg(node()->OutputAt(0)->DtypeName())
        .call_inplace_end()
        .call_end()
        .op_end();
  }
};

class RelaxConvCodeGen : public RelaxOpCodeGen {
 public:
  RelaxConvCodeGen(const String& func_name, bool use_bias)
      : RelaxOpCodeGen(func_name), use_bias_(use_bias) {}

 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_weight_arg("weight")
        .op_list_arg<int>("strides")
        .op_list_arg<int>("padding")
        .op_list_arg<int>("dilation")
        .op_arg<int>("groups")
        .op_str_arg("data_layout")
        .op_str_arg("kernel_layout")
        .op_str_arg("out_layout")
        .op_str_arg("out_dtype")
        .op_end();
    if (use_bias_) {
      std::string out_layout_str;
      ICHECK(node()->GetAttr("out_layout", &out_layout_str));
      const auto& out_layout = tir::Layout(out_layout_str);
      Array<Integer> expand_shape;
      for (size_t i = 0; i < node()->OutputAt(0)->Ndim(); i++) {
        if (out_layout[i].name() == "C") {
          expand_shape.push_back(node()->OutputAt(0)->DimAt(i));
        } else {
          expand_shape.push_back(Integer(1));
        }
      }
      BuilderEmit(stack, IdxNode());
      stack.call_start("relax.op.reshape")
          .call_arg(IdxWeight("bias", true))
          .call_list_arg(expand_shape, "shape")
          .call_end("expand_bias");
      BuilderEmit(stack, "expand_bias");
      stack.call_start("relax.op.add")
          .call_arg(IdxNode())
          .call_arg("expand_bias")
          .call_end(IdxNode());
    }
  }

 private:
  bool use_bias_;
};

class RelaxCumsumCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxCumsumCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_arg<int>("axis").op_str_arg("dtype").op_end();
  }
};

class RelaxStridedSliceCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxStridedSliceCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_list_arg<int>("axes")
        .op_list_arg<int>("begin")
        .op_list_arg<int>("end")
        .op_list_arg<int>("strides")
        .op_end();
  }
};

class RelaxEmbeddingCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxEmbeddingCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    const auto& input = node()->InputAt(0);
    if (input->DtypeName() != "int32") {
      stack.op_start("relax.op.astype").op_input_arg().call_str_arg("int32").op_end(IdxInput());
      BuilderEmit(stack, IdxInput());
    }
    if (input->Ndim() > 1) {
      stack.op_start("relax.op.reshape")
          .op_input_arg()
          .call_list_arg(std::vector<int>{-1}, "shape")
          .op_end(IdxInput());
      BuilderEmit(stack, IdxInput());
    }
    stack.op_start().op_weight_arg("weight").op_input_arg().op_arg<int>("axis").op_end();
    if (input->Ndim() > 1) {
      BuilderEmit(stack, IdxNode());
      stack.op_start("relax.op.reshape")
          .op_output_arg()
          .call_list_arg(node()->OutputAt(0)->shape)
          .op_end();
    }
  }
};

class RelaxFullCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxFullCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_list_arg<int>("shape")
        .op_input_arg(0, "fill_value")
        .op_str_arg("dtype")
        .op_end();
  }
};

class RelaxGroupNormCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxGroupNormCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_weight_arg("gamma")
        .op_weight_arg("beta")
        .op_arg<int>("num_groups")
        .op_arg<int>("channel_axis")
        .op_list_arg<int>("axes")
        .op_arg<float>("epsilon")
        .op_arg<bool>("center")
        .op_arg<bool>("scale")
        .op_end();
  }
};

class RelaxLayerNormCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxLayerNormCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_weight_arg("gamma")
        .op_weight_arg("beta")
        .op_list_arg<int>("axes")
        .op_arg<float>("epsilon")
        .op_arg<bool>("center")
        .op_arg<bool>("scale")
        .op_end();
  }
};

class RelaxLinearCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxLinearCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_weight_arg("weight")
        .op_weight_arg("bias")
        .op_str_arg("out_dtype")
        .op_end();
  }
};

class RelaxMatmulCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxMatmulCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_inputs_arg(false).op_str_arg("out_dtype").op_end();
  }
};

class RelaxNllLossCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxNllLossCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_inputs_arg(false)
        .op_str_arg("reduction")
        .op_arg<int>("ignore_index")
        .op_end();
  }
};

class RelaxPool2dCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxPool2dCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start()
        .op_input_arg()
        .op_list_arg<int>("pool_size")
        .op_list_arg<int>("strides")
        .op_list_arg<int>("padding")
        .op_list_arg<int>("dilation")
        .op_arg<bool>("ceil_mode")
        .op_str_arg("layout")
        .op_str_arg("out_layout")
        .op_end();
  }
};

class RelaxReduceAxisCodeGen : public RelaxOpCodeGen {
 public:
  RelaxReduceAxisCodeGen(const String& func_name, bool as_list)
      : RelaxOpCodeGen(func_name), as_list_(as_list) {}

 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg();
    if (as_list_) {
      stack.op_list_arg<int>("axis");
    } else {
      stack.op_arg<int>("axis");
    }
    stack.op_arg<bool>("keepdims").op_end();
  }

 private:
  bool as_list_;
};

class RelaxReshapeCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxReshapeCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_list_arg<int>("shape").op_end();
  }
};

class RelaxResize2dCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxResize2dCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    // roi has forced to be float list
    Array<String> roi_list;
    std::vector<float> roi;
    ICHECK(node()->GetAttr("roi", &roi)) << "Failed to get roi as float list";
    for (const auto& r : roi) {
      roi_list.push_back("float(" + std::to_string(r) + ")");
    }
    stack.op_start()
        .op_input_arg()
        .call_inplace_start("relax.ShapeExpr")
        .op_list_arg<int>("size", "values")
        .call_inplace_end()
        .call_list_arg(roi_list)
        .op_str_arg("layout")
        .op_str_arg("method")
        .op_str_arg("coordinate_transformation_mode")
        .op_str_arg("rounding_method")
        .op_arg<float>("cubic_alpha")
        .op_arg<int>("cubic_exclude")
        .op_arg<float>("extrapolation_value")
        .op_str_arg("out_dtype")
        .op_end();
  }
};

class RelaxShapeCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxShapeCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_list_arg<int>("shape", "values").op_end();
  }
};

class RelaxSimpleCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxSimpleCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_inputs_arg(false).op_end();
  }
};

class RelaxSplitCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxSplitCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg();
    int sections;
    if (node()->GetAttr("indices_or_sections", &sections)) {
      stack.op_arg<int>("indices_or_sections");
    } else {
      stack.op_list_arg<int>("indices_or_sections");
    }
    stack.op_arg<int>("axis").op_end();
  }
};

class RelaxTupleCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxTupleCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final { stack.op_start().op_inputs_arg().op_end(); }
};

class RelaxTriCodeGen : public RelaxOpCodeGen {
  RELAX_OP_CODEGEN_METHODS(RelaxTriCodeGen)
 protected:
  void CodeGenBuild(RelaxOpCodeStack& stack) final {
    stack.op_start().op_input_arg().op_arg<int>("k").op_end();
  }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<RelaxOpCodeGen>>>
GetRelaxOpCodeGens() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<RelaxOpCodeGen>>>();
  if (!map->empty()) return map;
  // binary && unary ops
  map->emplace("abs", std::make_shared<RelaxSimpleCodeGen>("relax.op.abs"));
  map->emplace("acos", std::make_shared<RelaxSimpleCodeGen>("relax.op.acos"));
  map->emplace("acosh", std::make_shared<RelaxSimpleCodeGen>("relax.op.acosh"));
  map->emplace("add", std::make_shared<RelaxSimpleCodeGen>("relax.op.add"));
  map->emplace("asin", std::make_shared<RelaxSimpleCodeGen>("relax.op.asin"));
  map->emplace("asinh", std::make_shared<RelaxSimpleCodeGen>("relax.op.asinh"));
  map->emplace("atan", std::make_shared<RelaxSimpleCodeGen>("relax.op.atan"));
  map->emplace("atanh", std::make_shared<RelaxSimpleCodeGen>("relax.op.atanh"));
  map->emplace("bitwise_and", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_and"));
  map->emplace("bitwise_not", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_not"));
  map->emplace("bitwise_or", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_or"));
  map->emplace("bitwise_xor", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_xor"));
  map->emplace("ceil", std::make_shared<RelaxSimpleCodeGen>("relax.op.ceil"));
  map->emplace("cos", std::make_shared<RelaxSimpleCodeGen>("relax.op.cos"));
  map->emplace("cosh", std::make_shared<RelaxSimpleCodeGen>("relax.op.cosh"));
  map->emplace("divide", std::make_shared<RelaxSimpleCodeGen>("relax.op.divide"));
  map->emplace("exp", std::make_shared<RelaxSimpleCodeGen>("relax.op.exp"));
  map->emplace("equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.equal"));
  map->emplace("floor", std::make_shared<RelaxSimpleCodeGen>("relax.op.floor"));
  map->emplace("floor_divide", std::make_shared<RelaxSimpleCodeGen>("relax.op.floor_divide"));
  map->emplace("greater", std::make_shared<RelaxSimpleCodeGen>("relax.op.greater"));
  map->emplace("greater_equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.greater_equal"));
  map->emplace("less", std::make_shared<RelaxSimpleCodeGen>("relax.op.less"));
  map->emplace("less_equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.less_equal"));
  map->emplace("log", std::make_shared<RelaxSimpleCodeGen>("relax.op.log"));
  map->emplace("logical_and", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_and"));
  map->emplace("logical_or", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_or"));
  map->emplace("logical_xor", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_xor"));
  map->emplace("maximum", std::make_shared<RelaxSimpleCodeGen>("relax.op.maximum"));
  map->emplace("minimum", std::make_shared<RelaxSimpleCodeGen>("relax.op.minimum"));
  map->emplace("multiply", std::make_shared<RelaxSimpleCodeGen>("relax.op.multiply"));
  map->emplace("negative", std::make_shared<RelaxSimpleCodeGen>("relax.op.negative"));
  map->emplace("not_equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.not_equal"));
  map->emplace("power", std::make_shared<RelaxSimpleCodeGen>("relax.op.power"));
  map->emplace("round", std::make_shared<RelaxSimpleCodeGen>("relax.op.round"));
  map->emplace("rsqrt", std::make_shared<RelaxSimpleCodeGen>("relax.op.rsqrt"));
  map->emplace("sigmoid", std::make_shared<RelaxSimpleCodeGen>("relax.op.sigmoid"));
  map->emplace("sign", std::make_shared<RelaxSimpleCodeGen>("relax.op.sign"));
  map->emplace("sin", std::make_shared<RelaxSimpleCodeGen>("relax.op.sin"));
  map->emplace("sinh", std::make_shared<RelaxSimpleCodeGen>("relax.op.sinh"));
  map->emplace("square", std::make_shared<RelaxSimpleCodeGen>("relax.op.square"));
  map->emplace("sqrt", std::make_shared<RelaxSimpleCodeGen>("relax.op.sqrt"));
  map->emplace("subtract", std::make_shared<RelaxSimpleCodeGen>("relax.op.subtract"));
  map->emplace("tan", std::make_shared<RelaxSimpleCodeGen>("relax.op.tan"));
  map->emplace("tanh", std::make_shared<RelaxSimpleCodeGen>("relax.op.tanh"));

  // reduce axis ops
  map->emplace("argmax", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.argmax", false));
  map->emplace("argmin", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.argmin", false));
  map->emplace("max", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.max", true));
  map->emplace("min", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.min", true));
  map->emplace("mean", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.mean", true));
  map->emplace("sum", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.sum", true));
  map->emplace("prod", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.prod", true));
  map->emplace("std", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.std", true));

  // axis ops
  map->emplace("nn.log_softmax", std::make_shared<RelaxAxisCodeGen>("relax.op.nn.log_softmax"));
  map->emplace("nn.softmax", std::make_shared<RelaxAxisCodeGen>("relax.op.nn.softmax"));

  // math ops
  map->emplace("astype", std::make_shared<RelaxAstypeCodeGen>("relax.op.astype"));
  map->emplace("broadcast_to", std::make_shared<RelaxBroadcastToCodeGen>("relax.op.broadcast_to"));
  map->emplace("clip", std::make_shared<RelaxClipCodeGen>("relax.op.clip"));
  map->emplace("constant", std::make_shared<RelaxConstantCodeGen>("relax.Var"));
  map->emplace("cumsum", std::make_shared<RelaxCumsumCodeGen>("relax.op.cumsum"));
  map->emplace("strided_slice",
               std::make_shared<RelaxStridedSliceCodeGen>("relax.op.strided_slice"));
  map->emplace("expand_dims", std::make_shared<RelaxAxesCodeGen>("relax.op.expand_dims"));
  map->emplace("full", std::make_shared<RelaxFullCodeGen>("relax.op.full"));
  map->emplace("matmul", std::make_shared<RelaxMatmulCodeGen>("relax.op.linear_algebra.matmul"));
  map->emplace("permute_dims", std::make_shared<RelaxAxesCodeGen>("relax.op.permute_dims"));
  map->emplace("reshape", std::make_shared<RelaxReshapeCodeGen>("relax.op.reshape"));
  map->emplace("split", std::make_shared<RelaxSplitCodeGen>("relax.op.split"));
  map->emplace("squeeze", std::make_shared<RelaxAxesCodeGen>("relax.op.squeeze"));
  map->emplace("tril", std::make_shared<RelaxTriCodeGen>("relax.op.tril"));
  map->emplace("triu", std::make_shared<RelaxTriCodeGen>("relax.op.triu"));

  // nn ops
  map->emplace("nn.adaptive_avg_pool2d",
               std::make_shared<RelaxAdaptivePool2dCodeGen>("relax.op.nn.adaptive_avg_pool2d"));
  map->emplace("nn.avg_pool2d", std::make_shared<RelaxPool2dCodeGen>("relax.op.nn.avg_pool2d"));
  map->emplace("nn.batch_norm", std::make_shared<RelaxBatchNormCodeGen>("relax.op.nn.batch_norm"));
  map->emplace("nn.conv1d", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv1d", false));
  map->emplace("nn.conv2d", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv2d", false));
  map->emplace("nn.gelu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.gelu"));
  map->emplace("nn.group_norm", std::make_shared<RelaxGroupNormCodeGen>("relax.op.nn.group_norm"));
  map->emplace("nn.layer_norm", std::make_shared<RelaxLayerNormCodeGen>("relax.op.nn.layer_norm"));
  map->emplace("nn.max_pool2d", std::make_shared<RelaxPool2dCodeGen>("relax.op.nn.max_pool2d"));
  map->emplace("nn.nll_loss", std::make_shared<RelaxNllLossCodeGen>("relax.op.nn.nll_loss"));
  map->emplace("nn.relu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.relu"));
  map->emplace("nn.silu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.silu"));

  // image ops
  map->emplace("image.resize2d", std::make_shared<RelaxResize2dCodeGen>("relax.op.image.resize2d"));

  // special op
  map->emplace("shape", std::make_shared<RelaxShapeCodeGen>("relax.ShapeExpr"));
  map->emplace("tuple", std::make_shared<RelaxTupleCodeGen>("relax.Tuple"));
  map->emplace("get_item", std::make_shared<RelaxGetItemCodeGen>("relax.TupleGetItem"));

  // msc ops
  map->emplace("msc.attention", std::make_shared<RelaxAttentionCodeGen>("relax.op.nn.attention"));
  map->emplace("msc.conv1d_bias", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv1d", true));
  map->emplace("msc.conv2d_bias", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv2d", true));
  map->emplace("msc.embedding", std::make_shared<RelaxEmbeddingCodeGen>("relax.op.take"));
  map->emplace("msc.linear",
               std::make_shared<RelaxLinearCodeGen>("relax.op.linear_algebra.linear"));
  map->emplace("msc.linear_bias",
               std::make_shared<RelaxLinearCodeGen>("relax.op.linear_algebra.linear"));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
