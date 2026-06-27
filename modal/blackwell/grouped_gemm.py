# modified by @deciding

# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
[GROUPED_GEMM] Grouped GEMM (C_g = A_g * B_g for each group g) for Blackwell SM100 using CuTe DSL.
[GROUPED_GEMM] Each group can have distinct (M, N, K). Uses TMA + persistent warp-specialized kernel
[GROUPED_GEMM] with per-group online tensor map updates.
[GROUPED_GEMM] DENSE counterpart dense_gemm_persistent.py solves a single (M,N,K,L) problem.
[GROUPED_GEMM] Key distinction: TMA descriptors are updated at runtime per group switch.
[GROUPED_GEMM] Dense: descriptors are static, set once at compile time.
"""

# [GROUPED_GEMM] Imports differ from dense:
#   - No argparse, lru_cache (dense has them for CLI and compile caching)
#   - Adds torch, sm100_utils (cutlass.utils.blackwell_helpers), cutlass.torch
#   - No create_cute_tensor_for_fp8 (grouped does not support fp8 yet)
#   - Dense uses utils.sm100.* directly; grouped uses sm100_utils alias
import functools
from typing import List, Type, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch


class GroupedGemmKernel:
    # [GROUPED_GEMM] Constructor: DENSE takes use_tma_store(bool); GROUPED takes tensormap_update_mode(enum).
    # [GROUPED_GEMM] Grouped has extra class constants (reserved_smem_bytes, bytes_per_tensormap, etc.).
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.tensormap_update_mode = tensormap_update_mode
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )
        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # [GROUPED_GEMM] Extra barrier: synchronizes tensormap init between epilogue-warp-0 and TMA warp.
        # [GROUPED_GEMM] No equivalent in dense.
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * (len(self.epilog_warp_id) + 1),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    # [GROUPED_GEMM] Grouped computes cluster_tile_shape_mnk (cluster-level tile) — not present in dense.
    # [GROUPED_GEMM] Grouped calls _compute_stages with different signature (takes epi_tile, c_layout, no use_tma_store).
    # [GROUPED_GEMM] Grouped validates mbar + tensormap smem fit within reserved_smem_bytes — no equivalent in dense.
    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # [GROUPED_GEMM] cluster_tile_shape_mnk = cta_tile * cluster_shape. Used by GroupTileScheduler.
        # [GROUPED_GEMM] No equivalent in dense — dense's scheduler just tiles the output tensor C.
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        # [GROUPED_GEMM] Uses utils.compute_epilogue_tile_shape; dense uses utils.sm100.compute_epilogue_tile_shape
        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        # [GROUPED_GEMM] _compute_stages returns (acc, ab, epi). Dense returns (acc, ab, c_stage).
        # [GROUPED_GEMM] Dense takes use_tma_store and c_smem_layout; grouped always uses TMA store.
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )
        # [GROUPED_GEMM] Uses sm100_utils alias; dense uses utils.sm100.
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        # [GROUPED_GEMM] epi_smem_layout_staged (always exists). Dense: c_smem_layout_staged only when use_tma_store.
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_epi_stage
        )
        # [GROUPED_GEMM] Validates mbar + tensormap smem consumption against reserved_smem_bytes.
        # [GROUPED_GEMM] No equivalent in dense — dense has no tensormap overhead.
        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_epi_stage=self.num_epi_stage,
        )
        tensormap_smem_bytes = self._get_tensormap_smem_bytes(
            self.tensormap_update_mode
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + GroupedGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    # [GROUPED_GEMM] __call__ signature is completely different from dense.
    # [GROUPED_GEMM] Dense: (a, b, c, max_active_clusters, stream, epilogue_op)
    # [GROUPED_GEMM] Grouped passes per-group metadata as tensors (problem_shape_mnkl, strides_abc, tensor_address_abc)
    # [GROUPED_GEMM] plus tensormap_cute_tensor (buffer for runtime TMA descriptor updates) and total_num_clusters.
    # [GROUPED_GEMM] No epilogue_op parameter (always identity).
    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # [GROUPED_GEMM] _setup_attributes called before tiled_mma is re-created below.
        # [GROUPED_GEMM] Dense does the same order but creates tiled_mma once and passes it.
        self._setup_attributes()

        # [GROUPED_GEMM] Re-creates tiled_mma after _setup_attributes (same as dense pattern).
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # [GROUPED_GEMM] TMA atom A/B: no internal_type=TFloat32 for Float32 (dense has this tweak).
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # [GROUPED_GEMM] TMA atom C: always created (grouped always uses TMA store).
        # [GROUPED_GEMM] Dense: conditional on use_tma_store, otherwise passes raw C tensor.
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            self.epi_tile,
        )

        # [GROUPED_GEMM] _compute_grid takes total_num_clusters (not output tensor C).
        # [GROUPED_GEMM] Uses StaticPersistentGroupTileScheduler; dense uses StaticPersistentTileScheduler.
        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        # [GROUPED_GEMM] SharedStorage struct: dense allocates via smem.allocate_tensor() for sA/sB/sC separately.
        # [GROUPED_GEMM] Grouped defines a @cute.struct with:
        #   - tensormap_buffer (extra — for runtime TMA descriptor storage in SMEM mode)
        #   - ab_full_mbar_ptr + ab_empty_mbar_ptr (dense has only one barrier set via make_participants)
        #   - acc_full_mbar_ptr + acc_empty_mbar_ptr (dense has full-only)
        #   - tmem_dealloc_mbar, tmem_holding_buf
        #   - sA, sB, sC wrapped in Align<MemRange> (dense uses smem.allocate_tensor with byte_alignment param)
        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else GroupedGemmKernel.num_tensormaps
            * GroupedGemmKernel.bytes_per_tensormap
            // 8
        )

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # [GROUPED_GEMM] Dense: no tensormap_cute_tensor, group_count, problem_shape_mnkl, strides_abc,
        # [GROUPED_GEMM] or tensor_address_abc in the call. Dense passes epilogue_op as last arg.
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,  # [GROUPED_GEMM] min_blocks_per_mp=1; dense doesn't set this
        )

    # [GROUPED_GEMM] kernel: Dense takes tma_atom_c Optional + fallback raw C tensor mC_mnl.
    # [GROUPED_GEMM] Grouped kernel always takes tma_atom_c, no fallback. No epilogue_op parameter.
    # [GROUPED_GEMM] Extra params: group_count, problem_sizes_mnkl, strides_abc, ptrs_abc, tensormaps.
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # [GROUPED_GEMM] Prefetches C descriptor unconditionally (dense guards with if use_tma_store).
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # [GROUPED_GEMM] tensormap_a/b/c_smem_ptr: only used in SMEM update mode.
        # [GROUPED_GEMM] Dense has no tensormap buffer — TMA descriptors are static.
        tensormap_a_smem_ptr = None
        tensormap_b_smem_ptr = None
        tensormap_c_smem_ptr = None
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = tensormap_smem_ptr
            tensormap_b_smem_ptr = (
                tensormap_a_smem_ptr + GroupedGemmKernel.bytes_per_tensormap // 8
            )
            tensormap_c_smem_ptr = (
                tensormap_b_smem_ptr + GroupedGemmKernel.bytes_per_tensormap // 8
            )

        # [GROUPED_GEMM] AB pipeline: uses separate full+empty barrier arrays.
        # [GROUPED_GEMM] Dense uses PipelineTmaUmma.create().make_participants() which internally manages
        # [GROUPED_GEMM] one set of barriers. Grouped manually creates both sets for explicit control.
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # [GROUPED_GEMM] ACC pipeline: uses acc_full_mbar_ptr + acc_empty_mbar_ptr arrays.
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # [GROUPED_GEMM] sA/sB/sC allocated from SharedStorage struct via get_tensor().
        # [GROUPED_GEMM] Dense allocates each via smem.allocate_tensor() with byte_alignment=128, swizzle.
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # [GROUPED_GEMM] Multicast: grouped has ab_empty_mcast_mask (for empty-barrier multicast release)
        # [GROUPED_GEMM] and peer multicast masks for 2CTA mode (a_full_mcast_mask_peer, b_full_mcast_mask_peer).
        # [GROUPED_GEMM] Dense only has a_full_mcast_mask and b_full_mcast_mask (no empty, no peer).
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        acc_full_mcast_mask = None
        if cutlass.const_expr(use_2cta_instrs):
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
                | cutlass.Int16(
                    0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask
                )
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # [GROUPED_GEMM] Grouped tile scheduler + tensormap workspace init — no equivalent in dense.
        # [GROUPED_GEMM] StaticPersistentGroupTileScheduler: iterates groups, then tiles M/N within each.
        # [GROUPED_GEMM] Dense uses StaticPersistentTileScheduler (single grid over M/N/L).
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )
        tensormap_manager = utils.TensorMapManager(
            self.tensormap_update_mode, GroupedGemmKernel.bytes_per_tensormap
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_a_init_ptr = tensormap_a_smem_ptr
            tensormap_b_init_ptr = tensormap_b_smem_ptr
            tensormap_c_init_ptr = tensormap_c_smem_ptr
        else:
            tensormap_a_init_ptr = tensormap_a_ptr
            tensormap_b_init_ptr = tensormap_b_ptr
            tensormap_c_init_ptr = tensormap_c_ptr

        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params,
            bid,
            grid_dim,
            self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(),
            group_count,
            problem_sizes_mnkl,
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        # [GROUPED_GEMM] TMA warp: dense loops over k_tiles in a simple while loop.
        # [GROUPED_GEMM] Grouped iterates over groups via the scheduler:
        #   - On group switch: updates A/B tensor maps to point to new group's memory
        #   - Fences tensormap updates before issuing TMA loads
        #   - Passes tma_desc_ptr to cute.copy for runtime TMA descriptor selection
        # [GROUPED_GEMM] Dense has no per-group logic, no tensormap updates.
        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            if cutlass.const_expr(self.delegate_tensormap_ab_init == False):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.tma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.tma_warp_id
                )
            tensormap_init_done = cutlass.Boolean(False)
            last_group_idx = cutlass.Int32(-1)
            work_tile = initial_work_tile_info
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                if not is_k_tile_cnt_zero:
                    is_group_changed = cur_group_idx != last_group_idx
                    if is_group_changed:
                        real_tensor_a = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.a_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            0,
                        )
                        real_tensor_b = self.make_tensor_for_tensormap_update(
                            cur_group_idx,
                            self.b_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            ptrs_abc,
                            1,
                        )
                        if not tensormap_init_done:
                            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                                self.tensormap_ab_init_barrier.arrive_and_wait()
                            tensormap_manager.fence_tensormap_initialization()
                            tensormap_init_done = True
                        tensormap_manager.update_tensormap(
                            (real_tensor_a, real_tensor_b),
                            (tma_atom_a, tma_atom_b),
                            (tensormap_a_ptr, tensormap_b_ptr),
                            self.tma_warp_id,
                            (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                        )
                    mma_tile_coord_mnl = (
                        grouped_gemm_cta_tile_info.cta_tile_idx_m
                        // cute.size(tiled_mma.thr_id.shape),
                        grouped_gemm_cta_tile_info.cta_tile_idx_n,
                        0,
                    )
                    tAgA_slice = tAgA[
                        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                    ]
                    tBgB_slice = tBgB[
                        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                    ]
                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )
                    if is_group_changed:
                        tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_ptr)
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(
                            ab_producer_state, peek_ab_empty_status
                        )
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=a_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_a_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                ab_producer_state
                            ),
                            mcast_mask=b_full_mcast_mask,
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_b_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < cur_k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                                ab_producer_state
                            )
                else:
                    if not tensormap_init_done:
                        if cutlass.const_expr(self.delegate_tensormap_ab_init):
                            self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx
            ab_pipeline.producer_tail(ab_producer_state)

        # [GROUPED_GEMM] MMA warp: dense iterates a simple k_tile_cnt from local_tile shape.
        # [GROUPED_GEMM] Grouped computes cur_k_tile_cnt from problem_shape_k from the scheduler.
        # [GROUPED_GEMM] All MMA logic is under is_leader_cta guard (dense also does this).
        # [GROUPED_GEMM] No tensormap involvement in MMA warp (only A/B data from SMEM).
        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            work_tile = initial_work_tile_info
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_group_idx = work_tile.group_search_result.group_idx
                problem_shape_k = work_tile.group_search_result.problem_shape_k
                cur_k_tile_cnt = (
                    problem_shape_k + self.cluster_tile_shape_mnk[2] - 1
                ) // self.cluster_tile_shape_mnk[2]
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

        # [GROUPED_GEMM] Epilogue warp: massively different from dense.
        # [GROUPED_GEMM] Dense delegates to utils.gemm.sm100.epilogue_tma_store() / epilogue() utility functions.
        # [GROUPED_GEMM] Grouped inlines the entire pipeline: tmem→rmem→smem→gmem with explicit steps:
        #   - delegate_tensormap_ab_init: if SMEM mode, epilogue warp 0 initializes A/B tensor maps in SMEM
        #     (arrive_and_wait signals TMA warp they're ready)
        #   - Initializes C tensor map from atom (not in dense — no C tensormap needed)
        #   - Allocates tmem, partitions copy atoms for tmem→rmem, rmem→smem, smem→gmem
        #   - On group switch: updates C tensor map via tensormap_manager.update_tensormap()
        #   - Per subtile: load from tmem, convert, store to smem, sync, TMA store to gmem
        #   - Handles zero-k-tile groups (fills C with zeros)
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.epilog_warp_id[0]
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.epilog_warp_id[0]
                )
                self.tensormap_ab_init_barrier.arrive_and_wait()
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_init_ptr,
                self.epilog_warp_id[0],
            )
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC)

            work_tile = initial_work_tile_info
            tensormap_manager.fence_tensormap_initialization()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_epi_stage,
                producer_group=c_producer_group,
            )
            # [GROUPED_GEMM] Epilogue main loop: per-group C tensor map updates + subtile writeback.
            # [GROUPED_GEMM] Dense's epilogue_tma_store() handles the entire writeback in one utility call
            # [GROUPED_GEMM] with no group iteration, no tensormap updates.
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    real_tensor_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,
                    )
                    tensormap_manager.update_tensormap(
                        ((real_tensor_c),),
                        ((tma_atom_c),),
                        ((tensormap_c_ptr),),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )
                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_epi_stage
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    if not is_k_tile_cnt_zero:
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
                        tRS_rC.fill(0)
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                if not is_k_tile_cnt_zero:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()

    # [GROUPED_GEMM] make_tensor_for_tensormap_update: no equivalent in dense.
    # [GROUPED_GEMM] Constructs a CuTe tensor for a specific group's A/B/C by reading
    # [GROUPED_GEMM] the pointer and strides from the per-group metadata tensors.
    # [GROUPED_GEMM] Used only for updating TMA descriptors — not for actual data movement.
    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )
        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        if cutlass.const_expr(tensor_index == 0):
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    # [GROUPED_GEMM] epilog_tmem_copy_and_partition: inlined in dense as part of
    # [GROUPED_GEMM] utils.gemm.sm100.epilogue_tma_store(). Grouped breaks it into explicit
    # [GROUPED_GEMM] separate methods for tmem→rmem, rmem→smem, smem→gmem partitioning.
    def epilog_tmem_copy_and_partition(
        self,
        tidx,
        tAcc,
        gC_mnl,
        epi_tile,
        use_2cta_instrs,
    ):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    # [GROUPED_GEMM] epilog_smem_copy_and_partition: dense handles rmem→smem copy inside epilogue utility.
    # [GROUPED_GEMM] Grouped separates it into its own method for clarity.
    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r,
        tTR_rC,
        tidx,
        sC,
    ):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tma_atom_c,
        gC_mnl,
        epi_tile,
        sC,
    ):
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    # [GROUPED_GEMM] _compute_stages: different signature from dense.
    # [GROUPED_GEMM] Grouped takes epi_tile, c_layout (no use_tma_store — always TMA store).
    # [GROUPED_GEMM] Uses sm100_utils.make_* instead of utils.sm100.make_*.
    # [GROUPED_GEMM] Accounts for reserved_smem_bytes (mbar + tensormap overhead).
    # [GROUPED_GEMM] Dense's version doesn't subtract tensormap overhead.
    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        c_layout,
        smem_capacity,
        occupancy,
    ):
        num_acc_stage = 2
        num_epi_stage = 2
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage
        num_ab_stage = (
            smem_capacity // occupancy
            - GroupedGemmKernel.reserved_smem_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        remaining_smem = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (GroupedGemmKernel.reserved_smem_bytes + epi_bytes)
        )
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    # [GROUPED_GEMM] _compute_grid: takes total_num_clusters (not output tensor C).
    # [GROUPED_GEMM] Uses StaticPersistentGroupTileScheduler; dense uses StaticPersistentTileScheduler.
    # [GROUPED_GEMM] The grid is 3D (cluster_m, cluster_n, num_cluster_groups) instead of
    # [GROUPED_GEMM] being derived from C's dimensions.
    @staticmethod
    def _compute_grid(
        total_num_clusters,
        cluster_shape_mn,
        max_active_clusters,
    ):
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )
        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    # [GROUPED_GEMM] _get_mbar_smem_bytes: not in dense. Calculates smem needed for barrier arrays
    # [GROUPED_GEMM] (full + empty for each of acc, ab, epi stages, 8 bytes per barrier).
    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages):
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        return sum(
            num_barriers_per_stage * num_bytes_per_barrier * stage
            for stage in kwargs_stages.values()
        )

    # [GROUPED_GEMM] _get_tensormap_smem_bytes: not in dense. Returns smem needed for tensormap
    # [GROUPED_GEMM] buffers (0 for GMEM mode, 3×128 bytes for SMEM mode).
    @staticmethod
    def _get_tensormap_smem_bytes(tensormap_update_mode):
        if tensormap_update_mode == utils.TensorMapUpdateMode.GMEM:
            return 0
        elif tensormap_update_mode == utils.TensorMapUpdateMode.SMEM:
            return (
                GroupedGemmKernel.bytes_per_tensormap * GroupedGemmKernel.num_tensormaps
            )
        else:
            raise ValueError(f"Invalid tensormap update mode: {tensormap_update_mode}")

    # [GROUPED_GEMM] _compute_num_tmem_alloc_cols: single-arg version.
    # [GROUPED_GEMM] Dense takes an extra `arch` parameter ("sm_100").
    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    # [GROUPED_GEMM] Class-level constants: not in dense (dense has no tensormap or barrier overhead).
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 3
    tensor_memory_management_bytes = 12


# [GROUPED_GEMM] create_tensor_and_stride: different from dense's prepare_tensors().
# [GROUPED_GEMM] Dense uses create_cute_tensor_for_fp8 with fp8 support; grouped uses cutlass_torch directly.
# [GROUPED_GEMM] Grouped bundles creation of torch_tensor, cute_tensor, cpu_fp32_ref, stride, and ptr.
# [GROUPED_GEMM] Dense's prepare_tensors() separates fp32 refs from storage tensors.
def create_tensor_and_stride(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    is_dynamic_layout: bool = True,
    torch_tensor_cpu: torch.Tensor = None,
) -> tuple[int, torch.Tensor, cute.Tensor, torch.Tensor, tuple[int, int]]:
    if torch_tensor_cpu is None:
        torch_tensor_cpu = cutlass_torch.matrix(l, mode0, mode1, is_mode0_major, dtype)
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu, dtype, is_dynamic_layout, assumed_align=16
    )
    return (
        torch_tensor.data_ptr(),
        torch_tensor,
        cute_tensor,
        torch_tensor_cpu,
        torch_tensor.stride()[:-1],
    )


# [GROUPED_GEMM] create_tensors_for_all_groups: no equivalent in dense.
# [GROUPED_GEMM] Creates tensors, strides, and pointers for all groups at once.
# [GROUPED_GEMM] Returns metadata arrays indexed by group_idx: ptrs_abc, strides_abc,
# [GROUPED_GEMM] cute_tensors_abc, and fp32 reference tensors for verification.
# [GROUPED_GEMM] Dense creates tensors for a single problem via prepare_tensors().
def create_tensors_for_all_groups(
    problem_sizes_mnkl,
    ab_dtype,
    c_dtype,
    a_major,
    b_major,
    c_major,
    torch_fp32_tensors_abc=None,
):
    if torch_fp32_tensors_abc is not None and len(torch_fp32_tensors_abc) != len(
        problem_sizes_mnkl
    ):
        raise ValueError("torch_fp32_tensors_abc must have one entry per group")
    new_torch_fp32_tensors_abc = (
        [] if torch_fp32_tensors_abc is None else torch_fp32_tensors_abc
    )
    torch_tensors_abc = []
    cute_tensors_abc = []
    strides_abc = []
    ptrs_abc = []
    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        existing_cpu_a = (
            torch_fp32_tensors_abc[group_idx][0] if torch_fp32_tensors_abc else None
        )
        existing_cpu_b = (
            torch_fp32_tensors_abc[group_idx][1] if torch_fp32_tensors_abc else None
        )
        existing_cpu_c = (
            torch_fp32_tensors_abc[group_idx][2] if torch_fp32_tensors_abc else None
        )
        (ptr_a, torch_tensor_a, cute_tensor_a, tensor_fp32_a, stride_mk_a) = (
            create_tensor_and_stride(
                l, m, k, a_major == "m", ab_dtype, torch_tensor_cpu=existing_cpu_a
            )
        )
        (ptr_b, torch_tensor_b, cute_tensor_b, tensor_fp32_b, stride_nk_b) = (
            create_tensor_and_stride(
                l, n, k, b_major == "n", ab_dtype, torch_tensor_cpu=existing_cpu_b
            )
        )
        (ptr_c, torch_tensor_c, cute_tensor_c, tensor_fp32_c, stride_mn_c) = (
            create_tensor_and_stride(
                l, m, n, c_major == "m", c_dtype, torch_tensor_cpu=existing_cpu_c
            )
        )
        if torch_fp32_tensors_abc is None:
            new_torch_fp32_tensors_abc.append(
                [tensor_fp32_a, tensor_fp32_b, tensor_fp32_c]
            )
        ptrs_abc.append([ptr_a, ptr_b, ptr_c])
        torch_tensors_abc.append([torch_tensor_a, torch_tensor_b, torch_tensor_c])
        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        cute_tensors_abc.append((cute_tensor_a, cute_tensor_b, cute_tensor_c))
    return (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        new_torch_fp32_tensors_abc,
    )


# [GROUPED_GEMM] run(): completely different from dense's run().
# [GROUPED_GEMM] Dense takes a single mnkl tuple + benchmark flag; grouped takes:
#   - num_groups + problem_sizes_mnkl list
#   - host_problem_shape_available flag
#   - tensormap_update_mode
# [GROUPED_GEMM] Dense uses compile_bmm wrapper → bmm wrapper → PersistentDenseGemmKernel.__call__.
# [GROUPED_GEMM] Grouped compiles GroupedGemmKernel directly (no bmm wrapper, no can_implement).
# [GROUPED_GEMM] Reference check: dense uses torch.bmm(); grouped uses per-group torch.einsum().
# [GROUPED_GEMM] Grouped creates tensormap buffers, per-group metadata tensors (strides, ptrs, dims).
# [GROUPED_GEMM] No fp8 support (raises ValueError for non-fp16/bf16).
# [GROUPED_GEMM] Dense has use_tma_store, epilogue_op; grouped does not.
# [GROUPED_GEMM] Benchmarking: grouped benchmarks whenever iterations > 0; dense has benchmark bool.
def run(
    num_groups: int,
    problem_sizes_mnkl: tuple[int, int, int, int],
    host_problem_shape_available: bool,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    use_2cta_instrs: bool,
    tensormap_update_mode: utils.TensorMapUpdateMode,
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    print("Running Blackwell Grouped GEMM test with:")
    print(f"{num_groups} groups")
    for i, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        print(f"Group {i}: {m}x{n}x{k}x{l}")
    print(f"AB dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Mma Tiler (M, N): {mma_tiler_mn}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"2CTA MMA instructions: {'True' if use_2cta_instrs else 'False'}")
    print(f"Tensor map update mode: {tensormap_update_mode}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # [GROUPED_GEMM] Validation: grouped validates inline (not via can_implement method like dense).
    # [GROUPED_GEMM] Grouped only supports fp16/bf16 AB; dense supports fp8/int8/tf32 too.
    if ab_dtype not in {cutlass.Float16, cutlass.BFloat16}:
        raise ValueError(f"Skip unsupported ab_dtype {ab_dtype}")
    if c_dtype not in {cutlass.Float16, cutlass.BFloat16, cutlass.Float32}:
        raise ValueError(f"Skip unsupported c_dtype {c_dtype}")
    if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
        raise ValueError(f"Skip unsupported acc_dtype {acc_dtype}")
    if ab_dtype == cutlass.BFloat16 and acc_dtype == cutlass.Float16:
        raise ValueError("Skip invalid ab_dtype and acc_dtype combination")
    if not (
        (not use_2cta_instrs and mma_tiler_mn[0] in [64, 128])
        or (use_2cta_instrs and mma_tiler_mn[0] in [128, 256])
    ):
        raise ValueError(f"Skip invalid mma tiler M {mma_tiler_mn[0]}")
    if mma_tiler_mn[1] not in range(32, 257, 32):
        raise ValueError(f"Skip invalid mma tiler N {mma_tiler_mn[1]}")
    if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
        raise ValueError(
            f"cluster_shape_m need align with use_2cta_instrs config {cluster_shape_mn}"
        )
    is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
    if (
        cluster_shape_mn[0] * cluster_shape_mn[1] > 16
        or cluster_shape_mn[0] <= 0
        or cluster_shape_mn[1] <= 0
        or not is_power_of_2(cluster_shape_mn[0])
        or not is_power_of_2(cluster_shape_mn[1])
    ):
        raise ValueError(f"Skip invalid cluster shape {cluster_shape_mn}")

    def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
        major_mode_idx = 0 if is_mode0_major else 1
        num_major_elements = tensor_shape[major_mode_idx]
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    for m, n, k, l in problem_sizes_mnkl:
        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            raise ValueError("Skip invalid problem alignment")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        torch_fp32_tensors_abc,
    ) = create_tensors_for_all_groups(
        problem_sizes_mnkl, ab_dtype, c_dtype, a_major, b_major, c_major
    )
    alignment = 16
    min_ab_size = alignment * 8 // ab_dtype.width
    min_c_size = alignment * 8 // c_dtype.width
    initial_cute_tensors_abc = [
        create_tensor_and_stride(1, min_ab_size, min_ab_size, a_major == "m", ab_dtype)[
            2
        ],
        create_tensor_and_stride(1, min_ab_size, min_ab_size, b_major == "n", ab_dtype)[
            2
        ],
        create_tensor_and_stride(1, min_c_size, min_c_size, c_major == "m", c_dtype)[2],
    ]
    hardware_info = utils.HardwareInfo()
    sm_count = hardware_info.get_max_active_clusters(1)
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    num_tensormap_buffers = sm_count
    tensormap_shape = (
        num_tensormap_buffers,
        GroupedGemmKernel.num_tensormaps,
        GroupedGemmKernel.bytes_per_tensormap // 8,
    )
    tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )
    grouped_gemm = GroupedGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        tensormap_update_mode,
    )
    tensor_of_dim_size_mnkl, tensor_of_dim_size_mnkl_torch = (
        cutlass_torch.cute_tensor_like(
            torch.tensor(problem_sizes_mnkl, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
    )
    tensor_of_strides_abc, tensor_of_strides_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_abc, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )
    tensor_of_ptrs_abc, tensor_of_ptrs_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_abc, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    def compute_total_num_clusters(problem_sizes_mnkl, cluster_tile_shape_mn):
        total_num_clusters = 0
        for m, n, _, _ in problem_sizes_mnkl:
            num_clusters_mn = tuple(
                (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
            )
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)
        return total_num_clusters

    def compute_cluster_tile_shape(mma_tiler_mn, cluster_shape_mn, use_2cta_instrs):
        cta_tile_shape_mn = list(mma_tiler_mn)
        if use_2cta_instrs:
            cta_tile_shape_mn[0] = cta_tile_shape_mn[0] // 2
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cluster_tile_shape_mn = compute_cluster_tile_shape(
        mma_tiler_mn, cluster_shape_mn, use_2cta_instrs
    )
    if host_problem_shape_available:
        print("Problem shapes available on host and device")
        total_num_clusters = compute_total_num_clusters(
            problem_sizes_mnkl, cluster_tile_shape_mn
        )
    else:
        print("Problem shapes available only on device")
        total_num_clusters = max_active_clusters

    current_stream = cutlass_torch.default_stream()
    try:
        from cutlass import CUDA_VERSION

        opt_level = (
            3
            if CUDA_VERSION.major < 13
            or (CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1)
            else 2
        )
    except ImportError:
        opt_level = 3

    # [GROUPED_GEMM] compile: grouped compiles GroupedGemmKernel directly with metadata tensors
    # [GROUPED_GEMM] (num_groups, problem_sizes, strides, ptrs, total_clusters, tensormaps, max_clusters).
    # [GROUPED_GEMM] Dense uses compile_bmm() → cute.compile(bmm, gemm, a, b, c, ...) — adds a bmm wrapper layer.
    compiled_grouped_gemm = cute.compile(
        grouped_gemm,
        initial_cute_tensors_abc[0],
        initial_cute_tensors_abc[1],
        initial_cute_tensors_abc[2],
        num_groups,
        tensor_of_dim_size_mnkl,
        tensor_of_strides_abc,
        tensor_of_ptrs_abc,
        total_num_clusters,
        tensor_of_tensormap,
        max_active_clusters,
        current_stream,
        options=f"--opt-level {opt_level}",
    )

    # [GROUPED_GEMM] Correctness: grouped uses per-group torch.einsum("mkl,nkl->mnl").
    # [GROUPED_GEMM] Dense uses torch.bmm(a_f32, b_f32) — single batch matmul.
    if not skip_ref_check:
        compiled_grouped_gemm(
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            tensor_of_tensormap,
            current_stream,
        )
        for i, (a, b, c) in enumerate(torch_tensors_abc):
            ref = torch.einsum(
                "mkl,nkl->mnl",
                a.cpu().to(dtype=torch.float32),
                b.cpu().to(dtype=torch.float32),
            )
            print(f"checking group {i}")
            torch.testing.assert_close(
                c.cpu(),
                ref.to(cutlass_torch.dtype(c_dtype)),
                atol=tolerance,
                rtol=1e-05,
            )

    # [GROUPED_GEMM] Benchmarking: grouped benchmarks whenever iterations > 0.
    # [GROUPED_GEMM] Dense uses a `benchmark` bool flag to control this.
    if iterations <= 0:
        return 0

    # [GROUPED_GEMM] Benchmark generator: creates fresh tensors + tensor maps per iteration
    # [GROUPED_GEMM] (dense's generator only creates a, b, c with no metadata or tensormap arrays).
    def generate_tensors():
        (
            ptrs_abc_workspace,
            torch_tensors_abc_workspace,
            cute_tensors_abc_workspace,
            strides_abc_workspace,
            _,
        ) = create_tensors_for_all_groups(
            problem_sizes_mnkl,
            ab_dtype,
            c_dtype,
            a_major,
            b_major,
            c_major,
            torch_fp32_tensors_abc,
        )
        initial_cute_tensors_abc_workspace = [
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, a_major == "m", ab_dtype
            )[2],
            create_tensor_and_stride(
                1, min_ab_size, min_ab_size, b_major == "n", ab_dtype
            )[2],
            create_tensor_and_stride(
                1, min_c_size, min_c_size, c_major == "m", c_dtype
            )[2],
        ]
        tensor_of_strides_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(strides_abc_workspace, dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensor_of_ptrs_abc_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc_workspace, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensormap_workspace, _ = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )
        args = testing.JitArguments(
            initial_cute_tensors_abc_workspace[0],
            initial_cute_tensors_abc_workspace[1],
            initial_cute_tensors_abc_workspace[2],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc_workspace,
            tensor_of_ptrs_abc_workspace,
            tensormap_workspace,
            current_stream,
        )
        args.add_to_scope([torch_tensors_abc_workspace])
        return args

    # [GROUPED_GEMM] Workspace size for cold L2: includes per-group tensors + strides + ptrs + tensormaps.
    # [GROUPED_GEMM] Dense only accounts for a_storage, b_storage, c_storage.
    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            sum(
                sum(
                    torch_tensor.numel() * torch_tensor.element_size()
                    for torch_tensor in group_tensors
                )
                for group_tensors in torch_tensors_abc
            )
            + tensor_of_strides_abc_torch.numel()
            * tensor_of_strides_abc_torch.element_size()
            + tensor_of_ptrs_abc_torch.numel() * tensor_of_ptrs_abc_torch.element_size()
            + tensor_of_tensormap_torch.numel()
            * tensor_of_tensormap_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_grouped_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    runtime_s = exec_time / 1.0e6
    total_flops = 2 * sum(M * N * K for (M, N, K, _) in problem_sizes_mnkl)
    print("Average Runtime : ", exec_time / 1000, "ms")
    print("GFLOPS          : ", total_flops / 1.0e9 / runtime_s)
    return exec_time
