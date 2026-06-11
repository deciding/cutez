# cutez.trace

`cutez.trace` is a GPU-side tracing toolkit for CuTe DSL kernels. Events are
recorded to a per-warp circular buffer in shared memory (SMEM) and flushed to
global memory (GMEM) at a point you choose (e.g. after the main loop). An
optional `disable_smem` mode writes events directly to GMEM, bypassing SMEM
entirely.

The host-side session allocates the GMEM output buffer, decodes the packed
64-bit event words, and produces Chrome/Perfetto trace JSON.

## Host-side setup: `CutezTraceSession`

```python
from cutez.trace.session import CutezTraceSession

session = CutezTraceSession(
    segments_per_block=int,        # required — number of warps per CTA (= number of trace segments)
    trace_path=str | Path,         # required — filesystem path for the output JSON
    block_available_bytes=int,     # required — SMEM bytes available per CTA (e.g. smem_capacity - fixed_allocations)
    total_blocks=int,              # optional — number of CTAs in the grid (default: 2)
    blocks_per_sm=int,             # optional — CTAs resident per SM for SMEM partitioning (default: 1)
    enabled=bool,                  # optional — disable tracing without removing instrumentation (default: True)
    verbose=bool,                  # optional — print SMEM diagnostics during kernel launch, only used when enabled=False (default: False)
    disable_smem=bool,             # optional — write events directly to GMEM, skip SMEM (default: False)
)
```

**`CutezTraceSession.__post_init__`** derives internal layout from
`block_available_bytes` / `blocks_per_sm` / `segments_per_block`:

- `block_smem_bytes = block_available_bytes // blocks_per_sm`
- block_smem_bytes is rounded down to a multiple of `segments_per_block * 8`
- `segment_bytes = block_smem_bytes // segments_per_block`
- The trace buffer address for block B, warp W is at offset
  `(B * segments_per_block + W) * segment_bytes` in the output tensor.

The session creates a **`TraceConfig`** which must be passed into the kernel for
`CutezTracer.create`.

### Key host-side methods

#### `session.write_trace_json(max_blocks: int | None = None) -> Path`

Decodes the session's buffer, pairs begin/end events, converts
`%clock` ticks to nanoseconds, and writes Chrome trace JSON to
`session.trace_path`. Call on the **host** after kernel completion.

- `max_blocks`: optional limit — only blocks 0..max_blocks-1 are included

#### `session.decode_buffer(words: torch.Tensor | None = None) -> dict[tuple[int, int], list]`

Decodes the flat GPU trace buffer into per-`(block, warp)` event lists.
Zero-valued slots are treated as unused and filtered out. Call on the **host**.
Only used if you want to debug the decoded buffer.

- `words`: optional explicit buffer; defaults to `session.buffer_tensor`

#### `session.reset_buffer()`

Zeroes the output buffer tensor. Call on the **host** between runs.

## GPU-side instrumentation: `CutezTracer`

Inside a `@cute.kernel` function, use `CutezTracer.create` to get a tracer
instance, then call `enter_scope` / `exit_scope` around the regions you want to
measure. Call `flush` after the last scope to push SMEM events into GMEM (or as
a no-op when `disable_smem=True`).

### `CutezTracer.create(out, seg_idx, smem, cfg, clock_ptr=None)`

Class method. Creates a tracer for one warp. Call **inside the kernel**.

| Parameter | Type | Description |
|-----------|------|-------------|
| `out` | `cute.Tensor` | The output tensor (e.g. `session.buffer` from host). Must be a 1-D `Int64` tensor with `buffer_numel` elements. |
| `seg_idx` | `cutlass.Int32` | The segment index for this warp. Typically `cute.arch.warp_idx()`. Must be warp-uniform. |
| `smem` | `SmemAllocator` | A `cutlass.utils.SmemAllocator` instance for SMEM allocation. |
| `cfg` | `TraceConfig` | The trace configuration from `session.trace_config`. |
| `clock_ptr` | optional | Pre-allocated SMEM pointer for the clock buffer; rarely needed. |

### `tracer.enter_scope(scope_id)` / `tracer.exit_scope(scope_id)`

Record a begin or end event. Call **inside the kernel**.

| Parameter | Type | Description |
|-----------|------|-------------|
| `scope_id` | `str` or `int` | Scope name (string) or numeric ID (0-255). Strings are interned to a unique ID on first use. Up to 255 distinct scope names are supported. |

### `tracer.flush()`

Copy all recorded events from the SMEM circular buffer to the warp's GMEM
segment. Only the warp leader thread (lane 0) performs the copy. When
`disable_smem=True` this is a no-op since events are already in GMEM. Call
**inside the kernel** after all `enter_scope`/`exit_scope` calls.

## Recording model

- Each warp gets a dedicated circular buffer segment of `segment_bytes`.
- Events are packed into 64-bit words: lower 32 bits = `%clock` low, upper 32
  bits = `%clock_hi` (11 bits), scope ID (8 bits), start/end flag (1 bit).
- The per-warp GMEM layout is the same regardless of `disable_smem`, so the
  host-side decoder reads events the same way.
- `%clock` is the GPU's free-running cycle counter, not wall-clock time. The
  session converts ticks to nanoseconds using `sm_clock_khz`.

## Example

```python
from pathlib import Path
from cutez.trace import CutezTraceSession
from cutez.trace import CutezTracer, TraceConfig

# ── Host side ────────────────────────────────────────────────────────────────
session = CutezTraceSession(
    segments_per_block=4,
    block_available_bytes=100352,
    trace_path="trace.json",
)
compiled = cute.compile(
    my_kernel, session.buffer, session.trace_config
)
compiled(session.buffer, session.trace_config)
torch.cuda.synchronize()
session.write_trace_json()
# Output: "Download trace.json and open in chrome://tracing"

# ── GPU kernel (inside @cute.kernel) ─────────────────────────────────────────
@cute.kernel
def my_kernel(out: cute.Tensor, trace_cfg: TraceConfig):
    wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = cutlass.utils.SmemAllocator()
    tracer = CutezTracer.create(out, smem=smem, seg_idx=wid, cfg=trace_cfg)

    tracer.enter_scope("load")
    # ... load data ...
    tracer.exit_scope("load")

    tracer.enter_scope("matmul")
    # ... compute ...
    tracer.exit_scope("matmul")

    cute.arch.sync_threads()
    tracer.flush()
```

## When to call what

| Call | Side | When |
|------|------|------|
| `CutezTraceSession(...)` | Host | Once, before kernel launch |
| `CutezTracer.create(...)` | Kernel | Once per warp, at the start of the kernel |
| `tracer.enter_scope(...)` | Kernel | Before each region of interest |
| `tracer.exit_scope(...)` | Kernel | After each region of interest |
| `tracer.flush()` | Kernel | After all scopes, before kernel exit (after `sync_threads`) |
| `session.write_trace_json()` | Host | After `torch.cuda.synchronize()` |
