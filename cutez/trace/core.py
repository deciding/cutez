"""Core SMEM trace-recording helpers for ``cutez_trace``.

This module packages the low-level CuTe DSL tracing primitives that were
previously provided through a top-level ``my_trace.py`` module. The API is kept
intentionally small and explicit so downstream repos can copy the entire
``cutez_trace`` directory without any extra root-level files.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

if not hasattr(cutlass, "int32"):
    cutlass.int32 = cutlass.Int32


@cute.jit
def init_clock(
    clock_ptr,
    out_ptr,
    seg_idx: cutlass.Int32,
    segment_size: cutlass.Int32,
):
    """Compute SMEM and GMEM segment pointers for one warp-owned trace segment."""
    seg_idx = cutlass.Int32(seg_idx)
    segment_size = cutlass.Int32(segment_size)
    smem_base_llvm = clock_ptr.llvm_ptr
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)

    wthreads = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 32)).result
    zero = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result

    smem_base_i32 = llvm.PtrToIntOp(i32, smem_base_llvm).result
    seg_base_off = llvm.MulOp(seg_idx, segment_size, 0).result
    seg_addr = llvm.AddOp(smem_base_i32, seg_base_off, 0).result

    gmem_base_llvm = out_ptr.llvm_ptr
    gmem_base_i64 = llvm.PtrToIntOp(i64, gmem_base_llvm).result
    seg_base_off_i64 = llvm.ZExtOp(i64, seg_base_off).result
    out_addr = llvm.AddOp(gmem_base_i64, seg_base_off_i64, 0).result

    tidx, _, _ = cute.arch.thread_idx()
    tidx_in_warp = llvm.URemOp(tidx, wthreads).result
    is_leader_thread = llvm.ICmpOp(llvm.ICmpPredicate.eq, tidx_in_warp, zero).result

    return seg_addr, out_addr, is_leader_thread


@dataclass
class CutezTracer:
    segment_size: int = 0
    seg_addr: object = None
    out_addr: object = None
    is_leader: object = None
    clock_idx: cutlass.Int32 = None

    @classmethod
    def create(
        cls,
        clock_ptr,
        out_ptr,
        seg_idx: cutlass.Int32,
        segment_size: int,
    ):
        seg_addr, out_addr, is_leader = init_clock(
            clock_ptr,
            out_ptr,
            seg_idx=seg_idx,
            segment_size=cutlass.Int32(segment_size),
        )
        return cls(
            segment_size=segment_size,
            seg_addr=seg_addr,
            out_addr=out_addr,
            is_leader=is_leader,
            clock_idx=cutlass.Int32(0),
        )

    def _record(self, is_start: bool, scope_id):
        is_leader = self.is_leader.ir_value()
        clock_record(
            is_start,
            scope_id,
            self.clock_idx,
            self.seg_addr,
            is_leader,
            cutlass.Int32(self.segment_size),
        )
        self.clock_idx += 1

    def enter_scope(self, scope_id):
        self._record(True, scope_id)

    def exit_scope(self, scope_id):
        self._record(False, scope_id)

    def flush(self):
        finanlize_clock(
            self.seg_addr,
            self.out_addr,
            cutlass.Int32(self.segment_size),
        )


@cute.jit
def clock_record(
    is_start: cutlass.Constexpr,
    scope_id: cutlass.Constexpr,
    clock_idx: cutlass.Int32,
    seg_addr: cutlass.Int32,
    is_leader_thread,
    segment_size: cutlass.Int32,
):
    """Record one begin/end event into the warp-owned SMEM circular segment."""
    clock_idx = cutlass.Int32(clock_idx)
    segment_size = cutlass.Int32(segment_size)

    if const_expr(is_start):
        scope_id = cutlass.int32((scope_id & 0xFF) << 23)
    else:
        scope_id = cutlass.int32(((scope_id & 0xFF) << 23) | (1 << 31))

    i32 = ir.IntegerType.get_signless(32)
    mask = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 0x7FF)).result
    entry_size = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 8)).result

    clock_off0 = llvm.MulOp(clock_idx, entry_size, 0).result
    clock_off = llvm.URemOp(clock_off0, segment_size).result
    smem_addr = llvm.AddOp(seg_addr, clock_off, 0).result

    clock_lo = llvm.inline_asm(
        i32,
        [],
        asm_string="mov.u32 $0, %clock;",
        constraints="=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    clock_hi = llvm.inline_asm(
        i32,
        [],
        asm_string="mov.u32 $0, %clock_hi;",
        constraints="=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    clock_hi = llvm.AndOp(clock_hi, mask).result
    clock_hi = llvm.OrOp(clock_hi, scope_id).result

    llvm.inline_asm(
        None,
        [smem_addr, clock_lo, clock_hi, is_leader_thread],
        asm_string="@$3 st.shared.v2.b32 [$0], {$1, $2};",
        constraints="r,r,r,b",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def finanlize_clock(
    seg_addr,
    out_addr,
    segment_size: cutlass.Int32,
):
    """Flush one SMEM trace segment to its matching GMEM segment."""
    segment_size = cutlass.Int32(segment_size)

    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    four = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 4)).result
    four_i64 = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 4)).result
    zero = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result
    zero_i64 = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 0)).result
    entry_size = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 8)).result
    entry_size_i64 = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 8)).result

    byte_off = zero
    byte_off_i64 = zero_i64
    clock_cnt = llvm.UDivOp(segment_size, entry_size).result

    for _ in cutlass.range(clock_cnt):
        smem_addr0 = llvm.AddOp(seg_addr, byte_off, 0).result

        loaded0 = llvm.inline_asm(
            i64,
            [smem_addr0],
            asm_string="ld.shared.b32 $0, [$1];",
            constraints="=l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        gmem_addr0 = llvm.AddOp(out_addr, byte_off_i64, 0).result
        llvm.inline_asm(
            None,
            [gmem_addr0, loaded0],
            asm_string="st.global.b32 [$0], $1;",
            constraints="l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        smem_addr1 = llvm.AddOp(smem_addr0, four, 0).result
        loaded1 = llvm.inline_asm(
            i64,
            [smem_addr1],
            asm_string="ld.shared.b32 $0, [$1];",
            constraints="=l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        gmem_addr1 = llvm.AddOp(gmem_addr0, four_i64, 0).result
        llvm.inline_asm(
            None,
            [gmem_addr1, loaded1],
            asm_string="st.global.b32 [$0], $1;",
            constraints="l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        byte_off = llvm.AddOp(byte_off, entry_size, 0).result
        byte_off_i64 = llvm.AddOp(byte_off_i64, entry_size_i64, 0).result


NUM_THREADS = 128


@cute.struct
class SharedStorage:
    """Shared-memory storage for one block's worth of raw 64-bit trace words."""

    clock_buf: cute.struct.MemRange[cutlass.Uint64, NUM_THREADS]
