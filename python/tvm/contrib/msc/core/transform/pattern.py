# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.core.transform.pattern"""


from tvm.relax.dpl.pattern import is_op, is_const, is_tuple_get_item, wildcard
from tvm.relax.transform import PatternCheckContext
from tvm.relax.backend.pattern_registry import register_patterns


def make_conv_bias_pattern(op_name):
    """A simple utility to create patterns for an operation fused with bias.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    input = wildcard()
    weight = is_const()
    conv = is_op(op_name)(input, weight)
    bias = is_const()
    shape = wildcard()
    reshape = is_op("relax.reshape")(bias, shape)
    out = is_op("relax.add")(conv, reshape)
    annotations = {"bias": bias, "reshape": reshape}
    return out, annotations


def _check_conv_bias(context: PatternCheckContext) -> bool:
    """Check if conv_bias fuse pattern is correct."""
    bias = context.annotated_expr["bias"]
    reshape = context.annotated_expr["reshape"]
    non_one_dims = len([i for i in reshape.struct_info.shape.values if i > 1])
    return non_one_dims <= 1 and bias.struct_info.ndim == 1


def make_linear_pattern():
    """A simple utility to create patterns for linear.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    input = wildcard()
    weight = is_const()
    permute = is_op("relax.permute_dims")(weight)
    out = is_op("relax.matmul")(input, permute)
    annotations = {"weight": weight, "permute": permute}
    return out, annotations


def _check_linear(context: PatternCheckContext) -> bool:
    """Check if linear pattern is correct."""
    weight = context.annotated_expr["weight"]
    permute = context.annotated_expr["permute"]
    return weight.struct_info.ndim == 2 and not permute.attrs["axes"]


def make_linear_bias_pattern():
    """A simple utility to create patterns for linear with bias.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    linear, annotations = make_linear_pattern()
    bias = is_const()
    out = is_op("relax.add")(linear, bias)
    annotations.update({"bias": bias, "out": out})
    return out, annotations


def _check_linear_bias(context: PatternCheckContext) -> bool:
    """Check if linear_bias pattern is correct."""
    if not _check_linear(context):
        return False
    bias = context.annotated_expr["bias"]
    return bias.struct_info.ndim == 1


def make_batch_norm_pattern():
    """A simple utility to create patterns for batch_norm.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    input = wildcard()
    gamma = is_const()
    beta = is_const()
    mean = is_const()
    var = is_const()
    batch_norm = is_op("relax.nn.batch_norm")(input, gamma, beta, mean, var)
    out = is_tuple_get_item(batch_norm)
    annotations = {"gamma": gamma, "beta": beta, "mean": mean, "var": var}
    return out, annotations


def _check_batch_norm(context: PatternCheckContext) -> bool:
    """Check if batch_norm pattern is correct."""
    """
    gamma = context.annotated_expr["gamma"]
    beta = context.annotated_expr["beta"]
    return gamma.struct_info.ndim == 1 and beta.struct_info.ndim == 1
    """
    return True


def make_embedding_pattern():
    """A simple utility to create patterns for 1d embedding.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    weight = is_const()
    input = wildcard()
    astype = is_op("relax.astype")(input)
    out = is_op("relax.take")(weight, astype)
    annotations = {"weight": weight, "astype": astype}
    return out, annotations


def _check_embedding(context: PatternCheckContext) -> bool:
    """Check if 1d embedding pattern is correct."""
    weight = context.annotated_expr["weight"]
    astype = context.annotated_expr["astype"]
    return (
        astype.attrs["dtype"] == "int32"
        and weight.struct_info.ndim == 2
        and weight.struct_info.dtype == "float32"
    )


def make_reshape_embedding_pattern():
    """A simple utility to create patterns for reshaped embedding.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    weight = is_const()
    input = wildcard()
    astype = is_op("relax.astype")(input)
    reduce_shape = wildcard()
    reduce_in = is_op("relax.reshape")(astype, reduce_shape)
    take = is_op("relax.take")(weight, reduce_in)
    expand_shape = wildcard()
    out = is_op("relax.reshape")(take, expand_shape)
    annotations = {"weight": weight, "astype": astype, "reduce_in": reduce_in}
    return out, annotations


def _check_reshape_embedding(context: PatternCheckContext) -> bool:
    """Check if reshape embedding pattern is correct."""
    weight = context.annotated_expr["weight"]
    if weight.struct_info.ndim != 2 or weight.struct_info.dtype != "float32":
        return False
    astype = context.annotated_expr["astype"]
    reduce_in = context.annotated_expr["reduce_in"]
    if astype.attrs["dtype"] != "int32" or reduce_in.struct_info.ndim != 1:
        return False
    return True


def make_attention_pattern():
    """A simple utility to create patterns for attention.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    q = wildcard()
    k = wildcard()
    v = wildcard()
    q_trans = is_op("relax.permute_dims")(q)
    k_trans = is_op("relax.permute_dims")(k)
    v_trans = is_op("relax.permute_dims")(v)
    out = is_op("relax.nn.attention")(q_trans, k_trans, v_trans)
    annotations = {"q_trans": q_trans, "k_trans": k_trans, "v_trans": v_trans}
    return out, annotations


def _check_attention(context: PatternCheckContext) -> bool:
    """Check if attention pattern is correct."""
    return True


def make_mask_attention_pattern():
    """A simple utility to create patterns for mask_attention.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """

    q = wildcard()
    k = wildcard()
    v = wildcard()
    mask = wildcard()
    q_trans = is_op("relax.permute_dims")(q)
    k_trans = is_op("relax.permute_dims")(k)
    v_trans = is_op("relax.permute_dims")(v)
    out = is_op("relax.nn.attention_bias")(q_trans, k_trans, v_trans, mask)
    annotations = {"q_trans": q_trans, "k_trans": k_trans, "v_trans": v_trans}
    return out, annotations


def _check_mask_attention(context: PatternCheckContext) -> bool:
    """Check if mask_attention pattern is correct."""
    return True


register_patterns(
    [
        (
            "msc.conv1d_bias",
            *make_conv_bias_pattern(
                "relax.nn.conv1d",
            ),
            _check_conv_bias,
        ),
        (
            "msc.conv2d_bias",
            *make_conv_bias_pattern(
                "relax.nn.conv2d",
            ),
            _check_conv_bias,
        ),
        (
            "msc.linear",
            *make_linear_pattern(),
            _check_linear,
        ),
        (
            "msc.linear_bias",
            *make_linear_bias_pattern(),
            _check_linear_bias,
        ),
        (
            "msc.embedding",
            *make_embedding_pattern(),
            _check_embedding,
        ),
        (
            "msc.embedding",
            *make_reshape_embedding_pattern(),
            _check_reshape_embedding,
        ),
        (
            "msc.attention",
            *make_attention_pattern(),
            _check_attention,
        ),
        (
            "msc.attention",
            *make_mask_attention_pattern(),
            _check_mask_attention,
        ),
    ]
)
