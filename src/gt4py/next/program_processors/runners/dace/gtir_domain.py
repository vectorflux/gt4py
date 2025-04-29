# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TypeAlias

import dace
import sympy

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.program_processors.runners.dace import gtir_sdfg_utils


FieldopDomain: TypeAlias = list[
    tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]
]
"""
Domain of a field operator represented as a list of tuples with 3 elements:
 - dimension definition
 - symbolic expression for lower bound (inclusive)
 - symbolic expression for upper bound (exclusive)
"""


def extract_domain(node: gtir.Node) -> FieldopDomain:
    """
    Visits the domain of a field operator and returns a list of dimensions and
    the corresponding lower and upper bounds. The returned lower bound is inclusive,
    the upper bound is exclusive: [lower_bound, upper_bound[
    """

    domain = []

    if cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")):
        for named_range in node.args:
            assert cpm.is_call_to(named_range, "named_range")
            assert len(named_range.args) == 3
            axis = named_range.args[0]
            assert isinstance(axis, gtir.AxisLiteral)
            lower_bound, upper_bound = (
                gtir_sdfg_utils.get_symbolic(arg) for arg in named_range.args[1:3]
            )
            dim = gtx_common.Dimension(axis.value, axis.kind)
            domain.append((dim, lower_bound, upper_bound))

    elif isinstance(node, domain_utils.SymbolicDomain):
        assert str(node.grid_type) in {"cartesian_domain", "unstructured_domain"}
        for dim, drange in node.ranges.items():
            domain.append(
                (
                    dim,
                    gtir_sdfg_utils.get_symbolic(drange.start),
                    gtir_sdfg_utils.get_symbolic(drange.stop),
                )
            )

    else:
        raise ValueError(f"Invalid domain {node}.")

    return domain


class GTIRDomainParser:
    domain_constraints: set[
        tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType, sympy.Basic]
    ]

    def __init__(self, domain: FieldopDomain):
        self.domain_constraints = {
            (lb, ub, sympy.var(f"__gtir_{dim.value}_size", integer=True, negative=False))
            for dim, lb, ub in domain
        }

    def simplify(self, expr: dace.symbolic.SymbolicType) -> dace.symbolic.SymbolicType:
        for lb, ub, size in self.domain_constraints:
            expr = expr.subs(lb, ub - size).subs(size, ub - lb)
        return expr
