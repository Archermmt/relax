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
"""tvm.contrib.msc.core.utils.expr_utils"""

from tvm import relax
from tvm.relax import PyExprVisitor
from tvm.relax.expr import VarBinding, Constant, Var
from tvm.contrib.msc.core import _ffi_api


def get_span_attrs(mod, entry_name="main"):
    """Extract the span attributes from relax.Function.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    entry_name: string
        The entry name of relax.Function.

    Returns
    -------
    attrs: dict
    """

    @relax.expr_functor.visitor
    class SpanVisitor(PyExprVisitor):
        def extract(self, expr):
            self._span_info = {}
            if isinstance(expr, relax.Expr):
                self.visit_expr(expr)
            elif isinstance(expr, relax.BindingBlock):
                self.visit_binding_block(expr)
            return self._span_info

        def _update_attrs(self, expr, name=None):
            if not expr.span:
                return
            name = name or _ffi_api.SpanGetAttr(expr.span, "name")
            if not name:
                return
            self._span_info[name] = _ffi_api.SpanGetAttrs(expr.span)

        def visit_var_binding_(self, binding: VarBinding) -> None:
            super().visit_var_binding_(binding)
            self._update_attrs(binding.value, binding.var.name_hint)

        def visit_constant_(self, op: Constant) -> None:
            super().visit_constant_(op)
            self._update_attrs(op)

        def visit_var_(self, op: Var) -> None:
            super().visit_var_(op)
            self._update_attrs(op, op.name_hint)

    return SpanVisitor().extract(mod[entry_name])


def show_function(mod, entry_name="main"):
    """Add span attrs after lines.

    Parameters
    ----------
    mod: IRModule
        The IRModule of relax.
    entry_name: string
        The entry name of relax.Function.

    Returns
    -------
    des: string
    """

    attrs, lines = get_span_attrs(mod, entry_name), []
    for l in str(mod[entry_name]).split("\n"):
        if ": " in l:
            v_name = l.strip().split(": ")[0]
            if v_name in attrs:
                l += (
                    " # "
                    + ", ".join(["{}={}".format(k, v) for k, v in attrs[v_name].items()])
                    + " #"
                )
        lines.append(l)
    return "\n".join(lines)
