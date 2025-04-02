# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
import os
from typing import Any, Sequence

import dace
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

from gt4py._core import definitions as core_defs
from gt4py.next import common, config, utils as gtx_utils
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners.dace import (
    sdfg_callable,
    transformations as gtx_transformations,
    utils as gtx_dace_utils,
    workflow as dace_worflow,
)


def _find_constant_symbols(
    sdfg: dace.SDFG,
    sdfg_call_args: dict[str, Any],
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, Any]:
    """Helper function to find symbols to replace with constant values."""
    include = {
        "istep",
        "limited_area",
        "lvn_only",
        "type_shear",
        "extra_diffu",
        "skip_compute_predictor_vertical_advection",
    }

    constant_symbols = gtx_transformations.gt_find_constant_arguments(
        sdfg_call_args,
        include=include,
    )

    if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE:
        # Search the stride symbols corresponding to the horizontal dimension
        for pname, arg in sdfg_call_args.items():
            if isinstance(arg, common.Field):
                dims = arg.domain.dims
                if len(dims) == 0:
                    continue
                elif len(dims) == 1:
                    dim_index = 0
                elif len(dims) == 2:
                    dim_index = 0 if dims[0].kind == common.DimensionKind.HORIZONTAL else 1
                else:
                    raise ValueError(f"Unsupported field with dims={dims}.")
                stride_name = gtx_dace_utils.field_stride_symbol_name(pname, dim_index)
                constant_symbols[stride_name] = 1
        # Same for connectivity tables, for which the first dimension is always horizontal
        for conn, desc in sdfg.arrays.items():
            if gtx_dace_utils.is_connectivity_identifier(conn, offset_provider_type):
                assert not desc.transient
                stride_name = gtx_dace_utils.field_stride_symbol_name(conn, 0)
                constant_symbols[stride_name] = 1

    return constant_symbols


def convert_args(
    inp: dace_worflow.compilation.CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    on_gpu = True if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM] else False

    def decorated_program(
        *args: Any,
        offset_provider: common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)
        flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
        if inp.implicit_domain:
            # generate implicit domain size arguments only if necessary
            size_args = arguments.iter_size_args(args)
            flat_size_args: Sequence[int] = gtx_utils.flatten_nested_tuple(tuple(size_args))
            flat_args = (*flat_args, *flat_size_args)

        sdfg = sdfg_program.sdfg
        if sdfg_program._lastargs:
            kwargs = dict(zip(sdfg.arg_names, flat_args, strict=True))
            kwargs.update(sdfg_callable.get_sdfg_conn_args(sdfg, offset_provider, on_gpu))

            use_fast_call = True
            last_call_args = sdfg_program._lastargs[0]
            # The scalar arguments should be overridden with the new value; for field arguments,
            # the data pointer should remain the same otherwise fast_call cannot be used and
            # the arguments list has to be reconstructed.
            for i, (arg_name, arg_type) in enumerate(inp.sdfg_arglist):
                if isinstance(arg_type, dace.data.Array):
                    assert arg_name in kwargs, f"argument '{arg_name}' not found."
                    data_ptr = get_array_interface_ptr(kwargs[arg_name], arg_type.storage)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    if last_call_args[i].value != data_ptr:
                        use_fast_call = False
                        break
                else:
                    assert isinstance(arg_type, dace.data.Scalar)
                    assert isinstance(last_call_args[i], ctypes._SimpleCData)
                    if arg_name in kwargs:
                        # override the scalar value used in previous program call
                        actype = arg_type.dtype.as_ctypes()
                        last_call_args[i] = actype(kwargs[arg_name])
                    else:
                        # shape and strides of arrays are supposed not to change, and can therefore be omitted
                        assert gtx_dace_utils.is_field_symbol(
                            arg_name
                        ), f"argument '{arg_name}' not found."

            if use_fast_call:
                return inp.fast_call()

        sdfg_args = sdfg_callable.get_sdfg_args(
            sdfg,
            offset_provider,
            *flat_args,
            check_args=False,
            on_gpu=on_gpu,
        )

        #####################  BEGIN  #####################
        if inp.sdfg is not None:
            # This is just a hack for jitting static symbols
            sdfg = inp.sdfg # we cannot recompile the SDFG, so we clone it
            inp.sdfg = None
            binary_filename = dace.codegen.compiler.get_binary_name(sdfg.build_folder, sdfg.name)
            if os.path.isfile(binary_filename):
                inp.sdfg_program = dace.codegen.compiler.load_from_file(sdfg, binary_filename)
            else:
                leading_kind = (
                    common.DimensionKind.HORIZONTAL
                    if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
                    else common.DimensionKind.VERTICAL
                )
                blocking_dim = (
                    "i_K_gtx_vertical" if config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE else None
                )
                offset_provider_type = common.offset_provider_to_type(offset_provider)
                constant_symbols = _find_constant_symbols(sdfg, sdfg_args, offset_provider_type)
                gtx_transformations.gt_auto_optimize(
                    sdfg,
                    gpu=on_gpu,
                    gpu_block_size=dace.Config.get("compiler", "cuda", "default_block_size"),
                    blocking_dim=blocking_dim,
                    blocking_size=10,
                    leading_kind=leading_kind,
                    constant_symbols=constant_symbols,
                    assume_pointwise=True,
                    make_persistent=False,
                )

                with dace.config.temporary_config():
                    dace.config.Config.set("compiler", "build_type", value=config.CMAKE_BUILD_TYPE.value)
                    # In some stencils, mostly in `apply_diffusion_to_w` the cuda codegen messes
                    #  up with the cuda streams, i.e. it allocates N streams but uses N+1.
                    #  This is a workaround until this issue if fixed in DaCe.
                    dace.config.Config.set("compiler", "cuda", "max_concurrent_streams", value=1)

                    if device == core_defs.DeviceType.CPU:
                        compiler_args = dace.config.Config.get("compiler", "cpu", "args")
                        # disable finite-math-only in order to support isfinite/isinf/isnan builtins
                        if "-ffast-math" in compiler_args:
                            compiler_args += " -fno-finite-math-only"
                        if "-ffinite-math-only" in compiler_args:
                            compiler_args.replace("-ffinite-math-only", "")

                        dace.config.Config.set("compiler", "cpu", "args", value=compiler_args)

                    inp.sdfg_program = sdfg.compile()
        #####################   END   #####################

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program
