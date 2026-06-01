# cutez_trace

`cutez_trace` is a small self-contained tracing toolkit for CuTe DSL kernels.

It provides:

- core SMEM trace-recording helpers in `cutez_trace.core`
- host-side output buffer allocation
- decode helpers for packed 64-bit event records
- Chrome/Perfetto trace JSON writing
- example kernels that show the intended manual instrumentation style

## Recording model

The low-level helpers live in `cutez_trace.core`:

- `init_clock(...)`
- `clock_record(...)`
- `finanlize_clock(...)`
- `SharedStorage`

User code is responsible for:

- using the same `wid` as the segment id
- maintaining `clock_idx`
- understanding that `clock_idx` wraps in the ring when it exceeds the segment capacity

`CutezTraceSession.write_trace_json(...)` converts raw GPU `%clock` ticks to
nanoseconds with the device SM clock rate reported by CUDA. That conversion is
approximate host-side scaling, not calibrated wall-clock timing.

The shipped example records all four warps in one 128-thread block. Each warp
emits one outer scope and repeated inner add scopes, so the ring buffer wraps
once `iters` is large enough.

## Example

```python
from pathlib import Path

from cutez.trace.examples import run_sample_trace

result = run_sample_trace(Path("trace.json"), iters=4)
print(result["trace_path"])
```

## Copying To Another Repo

The package is designed so you can copy the entire `cutez_trace/` directory as a
single unit. If you also want the packaged tests, copy `cutez_trace/tests/` as
part of the same folder.

## Viewing the trace

Open the generated `trace.json` in Perfetto or `chrome://tracing`.
