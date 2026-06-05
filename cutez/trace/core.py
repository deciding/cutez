"""Core SMEM trace-recording helpers for ``cutez_trace``.

This module packages the low-level CuTe DSL tracing primitives that were
previously provided through a top-level ``my_trace.py`` module. The API is kept
intentionally small and explicit so downstream repos can copy the entire
``cutez_trace`` directory without any extra root-level files.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Constexpr, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.core import const
from cutez._params_base import ParamsBase

if not hasattr(cutlass, "int32"):
    cutlass.int32 = cutlass.Int32


_REGION_TO_ID: dict[str, int] = {}
_ID_TO_REGION: dict[int, str] = {}


def _intern_region(name: str) -> int:
    region_id = _REGION_TO_ID.get(name)
    if region_id is not None:
        return region_id
    region_id = len(_REGION_TO_ID) + 1
    if region_id > 0xFF:
        raise ValueError("cutez.trace supports at most 255 distinct scope names")
    _REGION_TO_ID[name] = region_id
    _ID_TO_REGION[region_id] = name
    return region_id


def get_region_names() -> dict[int, str]:
    return dict(_ID_TO_REGION)


@cute.jit
def debug_smem_usage(smem_capacity_bytes: cutlass.Int32):
    """Print dynamic SMEM size and architectural SMEM capacity from device code.

    This is a standalone debug helper — it does not affect trace behavior.
    Call it from a kernel to inspect SMEM usage (only thread 0 prints).
    The caller is responsible for getting *smem_capacity_bytes* from the
    host-side ``get_smem_cap()`` helper and threading it through to device code.
    """
    tidx, _, _ = cute.arch.thread_idx()
    dyn = cute.arch.get_dyn_smem_size()
    bdx, bdy, bdz = cute.arch.block_dim()
    gdx, gdy, gdz = cute.arch.grid_dim()
    zero = cutlass.Int32(0)
    base_align = cutlass.Int32(1024)
    if tidx == zero:
        num_warps = bdx * bdy * bdz // 32
        grid_total = gdx * gdy * gdz
        available = smem_capacity_bytes - base_align - dyn
        cute.printf("dyn_smem_bytes=%d", dyn)
        cute.printf("capacity_bytes=%d", smem_capacity_bytes)
        cute.printf("available_bytes=%d", available)
        cute.printf("threads=%d", bdx * bdy * bdz)
        cute.printf("warps=%d", num_warps)
        cute.printf("grid=%dx%dx%d=%d", gdx, gdy, gdz, grid_total)


def get_smem_cap(compute_capability: str | None = None) -> int:
    """Get (and print) the SMEM capacity in bytes for the given compute capability.

    When *compute_capability* is ``None``, the current device arch is auto-detected.
    Use the returned value to call ``debug_smem_usage(smem_cap)`` from device code.
    """
    cap = cutlass.utils.get_smem_capacity_in_bytes(
        compute_capability=compute_capability
    )
    print(f"SMEM capacity ({compute_capability or 'auto'}): {cap} bytes")
    return cap


@cute.jit
def init_clock(
    clock_ptr,
    out_ptr,
    seg_idx: cutlass.Int32,
    segment_size: cutlass.Int32,
    block_smem_bytes: cutlass.Int32,
    total_blocks: cutlass.Int32,
):
    """Compute SMEM and GMEM segment pointers for one warp-owned trace segment."""
    seg_idx = cutlass.Int32(seg_idx)
    segment_size = cutlass.Int32(segment_size)
    block_smem_bytes = cutlass.Int32(block_smem_bytes)
    total_blocks = cutlass.Int32(total_blocks)
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
    bidx, bidy, bidz = cute.arch.block_idx()
    gdx, gdy, _ = cute.arch.grid_dim()
    block_xy = llvm.MulOp(bidy, gdx, 0).result
    block_z = llvm.MulOp(bidz, llvm.MulOp(gdx, gdy, 0).result, 0).result
    linear_block = llvm.AddOp(llvm.AddOp(bidx, block_xy, 0).result, block_z, 0).result
    block_smem_bytes_i64 = llvm.ZExtOp(i64, block_smem_bytes).result
    block_base_off_i64 = llvm.MulOp(
        llvm.ZExtOp(i64, linear_block).result, block_smem_bytes_i64, 0
    ).result
    seg_base_off_i64 = llvm.ZExtOp(i64, seg_base_off).result
    out_addr = llvm.AddOp(
        gmem_base_i64,
        llvm.AddOp(block_base_off_i64, seg_base_off_i64, 0).result,
        0,
    ).result

    tidx, _, _ = cute.arch.thread_idx()
    tidx_in_warp = llvm.URemOp(tidx, wthreads).result
    is_leader_thread = llvm.ICmpOp(llvm.ICmpPredicate.eq, tidx_in_warp, zero).result

    recording_block = llvm.ICmpOp(
        llvm.ICmpPredicate.ult, linear_block, total_blocks
    ).result

    return seg_addr, out_addr, is_leader_thread, recording_block


@dataclass
class TraceConfig(ParamsBase):
    block_smem_bytes: int
    segment_bytes: int
    smem_words: int
    dummy: bool = False
    smem_capacity_bytes: int = 0
    total_blocks: int = 2


@dataclass
class CutezTracer:
    segment_size: object = None
    seg_addr: object = None
    out_addr: object = None
    is_leader: object = None
    recording_block: object = None
    clock_idx: cutlass.Int32 = None
    dummy: Constexpr = const_expr(False)

    @classmethod
    def create(
        cls,
        out,
        seg_idx: cutlass.Int32,
        smem,
        cfg: TraceConfig,
        clock_ptr=None,
    ):
        if cfg.dummy:
            return cls(dummy=const_expr(True))

        if clock_ptr is None:
            clock_smem = smem.allocate_tensor(
                element_type=cutlass.Uint64,
                layout=cfg.smem_words,
                byte_alignment=8,
            )
            clock_ptr = clock_smem.iterator
        seg_addr, out_addr, is_leader, recording_block = init_clock(
            clock_ptr,
            out.iterator,
            seg_idx=seg_idx,
            segment_size=cutlass.Int32(cfg.segment_bytes),
            block_smem_bytes=cutlass.Int32(cfg.block_smem_bytes),
            total_blocks=cutlass.Int32(cfg.total_blocks),
        )
        segment_size = cutlass.Int32(cfg.segment_bytes)
        return cls(
            segment_size=segment_size,
            seg_addr=seg_addr,
            out_addr=out_addr,
            is_leader=is_leader,
            recording_block=recording_block,
            clock_idx=cutlass.Int32(0),
        )

    def _record(self, is_start: bool, scope_id):
        if const_expr(self.dummy):
            return
        if isinstance(scope_id, str):
            scope_id = _intern_region(scope_id)
        is_leader = self.is_leader.ir_value()
        clock_record(
            is_start,
            scope_id,
            self.clock_idx,
            self.seg_addr,
            is_leader,
            self.segment_size,
        )
        self.clock_idx += 1

    def enter_scope(self, scope_id):
        self._record(True, scope_id)

    def exit_scope(self, scope_id):
        self._record(False, scope_id)

    def flush(self):
        if const_expr(self.dummy):
            return
        recording_block = self.recording_block.ir_value()
        finanlize_clock(
            self.seg_addr,
            self.out_addr,
            self.segment_size,
            self.clock_idx,
            recording_block,
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
    # tidx, _, _ = cute.arch.thread_idx()
    # bidx, bidy, bidz = cute.arch.block_idx()
    # gdx, gdy, _ = cute.arch.grid_dim()
    # block_linear = bidx + gdx*bidy + gdx*gdy*bidz
    # if tidx == 160:
    #    cute.printf("{} clock_hi: {}", block_linear, clock_hi)
    #    cute.printf("{} clock_lo: {}", block_linear, clock_lo)
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
    num_events: cutlass.Int32,
    recording_block,
):
    """Flush recorded trace entries from SMEM segment to its matching GMEM segment."""
    segment_size = cutlass.Int32(segment_size)
    num_events = cutlass.Int32(num_events)

    num_events = cutlass.min(segment_size // 8, num_events)

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

    for _ in cutlass.range(num_events):
        smem_addr0 = llvm.AddOp(seg_addr, byte_off, 0).result

        loaded0 = llvm.inline_asm(
            i64,
            [smem_addr0, recording_block],
            asm_string="@$2 ld.shared.b32 $0, [$1];",
            constraints="=l,r,b",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        gmem_addr0 = llvm.AddOp(out_addr, byte_off_i64, 0).result
        llvm.inline_asm(
            None,
            [gmem_addr0, loaded0, recording_block],
            asm_string="@$2 st.global.b32 [$0], $1;",
            constraints="l,l,b",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        smem_addr1 = llvm.AddOp(smem_addr0, four, 0).result
        loaded1 = llvm.inline_asm(
            i64,
            [smem_addr1, recording_block],
            asm_string="@$2 ld.shared.b32 $0, [$1];",
            constraints="=l,r,b",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

        gmem_addr1 = llvm.AddOp(gmem_addr0, four_i64, 0).result
        llvm.inline_asm(
            None,
            [gmem_addr1, loaded1, recording_block],
            asm_string="@$2 st.global.b32 [$0], $1;",
            constraints="l,l,b",
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
