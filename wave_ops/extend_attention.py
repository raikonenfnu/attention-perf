# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1

import torch
import math

import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.constraints import MMAType

from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)

import os

dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))

def extend_attention_wave(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    b_req_idx,
    b_seq_len,
    b_seq_len_extend,
    b_start_loc_extend,
    max_seq_len,
    output,
    is_causal=True,
    layer_scaling=None,
    logit_cap=0,
):

    shape = AttentionShape(
        num_query_heads=q_extend.shape[1],
        num_kv_heads=k_extend.shape[1],
        head_size=q_extend.shape[2],
        head_size_kv=k_extend.shape[2],
        num_seqs=b_seq_len.shape[0],
        max_seq_len=max_seq_len,
        block_size=64,
    )

    assert shape.num_query_heads % shape.num_kv_heads == 0

    # Run the wave kernel.
    mfma_variant = (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16)
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        req_to_tokens.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        input_dtype=q_extend.dtype,
        output_dtype=output.dtype,
        size_dtype=b_seq_len.dtype,
        is_causal=is_causal,
        layer_scaling=layer_scaling,
        logit_cap=logit_cap,
    )

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()

    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}
    config["dump_intermediates"] = "./0218_perf"
    # config["print_ir_after_all"] = True
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        schedule=False,
        compile_config=compile_config,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        # TODO: Add scaling of QK as part of kernel.
        mb = extend_attention(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            output,
        )

        if dump_generated_mlir:
            shape_list = [
                q_extend.shape[0],
                q_extend.shape[1],
                k_extend.shape[1],
                q_extend.shape[2],
                k_extend.shape[2],
            ]
            filename = f"wave_prefill_attention_{'x'.join(map(str, shape_list))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())
