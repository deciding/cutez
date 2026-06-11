# cutez

## Install
```
pip install cutez
```

## Usage

### Autotune

```python
# Step 1: define the grid search scope
AUTOTUNE_CONFIGS = [
    cutez.Config(kwargs={"mma_tiler_mn": (256, 256), "cluster_shape_mn": (2, 1), "ab_stages": ab_stages,})
    for ab_stages in (6, 7, 8)
]

...

# Step 2: decorate the host function. can specify the persistent cache path
@cutez.autotune(configs=AUTOTUNE_CONFIGS, key=["m", "n", "k"],  cache_path='/workspace/dump/dense_gemm_7min.json')
@cute.jit
def host_function(
...


# Step 3: autotune requires compile with cutez.compile. use verbose=True to see the configs.
compiled_gemm = cutez.compile(
    ...
    verbose=True, # To show what is the best config
)
```

### Trace

`cutez.trace` provides GPU-side instrumentation for CuTe DSL kernels. Events
are recorded to per-warp circular buffers and flushed to global memory, then
decoded into Chrome/Perfetto trace JSON on the host.

```python
from cutez.trace import CutezTraceSession, CutezTracer, TraceConfig

# ── Host side ────────────────────────────────────────────────────────────────
session = CutezTraceSession(
    segments_per_block=4,
    block_available_bytes=100352,
    trace_path="trace.json",
)
compiled = cute.compile(my_kernel, session.buffer, session.trace_config)
compiled(session.buffer, session.trace_config)
torch.cuda.synchronize()
session.write_trace_json()

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

See [`cutez/trace/README.md`](cutez/trace/README.md) for the full API reference,
including `CutezTracer`, `TraceConfig`, optional arguments, and the recording
model.

## Build Wheel

Build the source distribution and wheel with:

```bash
python -m build --sdist --wheel
```
