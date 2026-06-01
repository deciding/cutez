"""Host helpers and examples for SMEM-backed cutez trace recording."""

from .format import (
    ChromeTraceEvent,
    PackedEvent,
    decode_packed_event,
    decode_ring_events,
    pair_complete_events,
    trace_json_payload,
)
from .session import CutezTraceSession

__all__ = [
    "ChromeTraceEvent",
    "TraceConfig",
    "CutezTracer",
    "CutezTraceSession",
    "get_region_names",
    "PackedEvent",
    "SharedStorage",
    "clock_record",
    "decode_packed_event",
    "decode_ring_events",
    "finanlize_clock",
    "init_clock",
    "pair_complete_events",
    "run_sample_trace",
    "trace_json_payload",
]


def __getattr__(name: str):
    """Lazily expose optional example and core helpers from the package root."""
    if name == "run_sample_trace":
        from .examples import run_sample_trace

        return run_sample_trace
    if name in {
        "TraceConfig",
        "CutezTracer",
        "SharedStorage",
        "clock_record",
        "get_region_names",
        "finanlize_clock",
        "init_clock",
    }:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
