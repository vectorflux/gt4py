# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Final, Iterable, Optional, Protocol, Sequence, TypeAlias

import dace
from dace import subsets as dace_subsets
import sympy

from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.ffront import fbuiltins as gtx_fbuiltins
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.program_processors.runners.dace import (
    gtir_dataflow,
    gtir_python_codegen,
    gtir_sdfg,
    gtir_sdfg_utils,
    utils as gtx_dace_utils,
)
from gt4py.next.program_processors.runners.dace.gtir_scan_translator import translate_scan
from gt4py.next.type_system import type_info as ti, type_specifications as ts


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

    domain_constraints: set[tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType, sympy.Basic]]
    
    def __init__(self, domain: FieldopDomain):
        self.domain_constraints = {
            (lb, ub, sympy.var(f"{dim.value}_size", positive=True))
            for dim, lb, ub in domain
        }
            
    def simplify(self, expr: dace.symbolic.SymbolicType) -> dace.symbolic.SymbolicType:
        for lb, ub, size in self.domain_constraints:
            expr = expr.subs(lb, ub - size).subs(size, ub - lb)
        return expr
