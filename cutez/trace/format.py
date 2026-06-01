"""Pure-Python decode helpers for SMEM-flushed cutez trace records.

This module knows how to decode the packed 64-bit words produced by
`my_trace.clock_record(...)` and convert them into Chrome trace events.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedEvent:
    word: int
    block: int
    warp: int
    logical_index: int
    clock_lo: int
    clock_hi_low11: int
    scope_id: int
    is_start: bool

    @property
    def raw_clock(self) -> int:
        return (self.clock_hi_low11 << 32) | self.clock_lo


@dataclass(frozen=True)
class ChromeTraceEvent:
    name: str
    ts: int
    dur: int
    pid: int
    tid: int
    block: int
    warp: int
    scope_id: int


def decode_packed_event(
    word: int, *, block: int, warp: int, logical_index: int
) -> PackedEvent:
    """Decode one packed 64-bit trace word into a host-side event record.

    The caller passes the raw event word plus its `(block, warp)` ownership and
    the monotonic logical index assigned during ring reconstruction. The return
    value is a `PackedEvent` with unpacked clock bits, scope id, and direction.
    """
    lo = word & 0xFFFFFFFF
    hi = (word >> 32) & 0xFFFFFFFF
    return PackedEvent(
        word=word,
        block=block,
        warp=warp,
        logical_index=logical_index,
        clock_lo=lo,
        clock_hi_low11=hi & 0x7FF,
        scope_id=(hi >> 23) & 0xFF,
        is_start=((hi >> 31) & 1) == 0,
    )


def decode_ring_events(words: list[int], *, block: int, warp: int) -> list[PackedEvent]:
    """Reconstruct retained events from one warp-owned ring segment.

    This decoder treats zero-valued entries as unused slots, decodes every
    nonzero entry in the segment, and orders the retained events by reconstructed
    clock value. It intentionally does not require caller-provided event counts.
    """
    decoded = [
        decode_packed_event(word, block=block, warp=warp, logical_index=i)
        for i, word in enumerate(words)
        if word != 0
    ]
    decoded.sort(key=lambda event: (event.raw_clock, event.logical_index))
    return [
        PackedEvent(
            word=event.word,
            block=event.block,
            warp=event.warp,
            logical_index=i,
            clock_lo=event.clock_lo,
            clock_hi_low11=event.clock_hi_low11,
            scope_id=event.scope_id,
            is_start=event.is_start,
        )
        for i, event in enumerate(decoded)
    ]


def pair_complete_events(
    events: list[PackedEvent], *, region_names: dict[int, str] | None = None
) -> list[ChromeTraceEvent]:
    """Pair begin/end records into Chrome complete events.

    The input must already be ordered per warp, typically from
    `decode_ring_events(...)` on a segment keyed by `wid`. The return value is a
    list of `ChromeTraceEvent` objects keyed by block, warp, and scope id, using
    `region_names` when provided.
    """
    region_names = region_names or {}
    stacks: dict[tuple[int, int, int], list[PackedEvent]] = {}
    out: list[ChromeTraceEvent] = []
    for event in events:
        key = (event.block, event.warp, event.scope_id)
        if event.is_start:
            stacks.setdefault(key, []).append(event)
            continue
        stack = stacks.get(key, [])
        if not stack:
            continue
        start = stack.pop()
        dur = max(0, event.raw_clock - start.raw_clock)
        out.append(
            ChromeTraceEvent(
                name=region_names.get(event.scope_id, str(event.scope_id)),
                ts=start.raw_clock,
                dur=dur,
                pid=event.block,
                tid=event.warp * 32,
                block=event.block,
                warp=event.warp,
                scope_id=event.scope_id,
            )
        )
    return out


def trace_json_payload(events: list[ChromeTraceEvent]) -> dict:
    if not events:
        return {"displayTimeUnit": "ns", "traceEvents": []}
    min_ts = min(event.ts for event in events)
    scale = 1e-3
    return {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {
                "name": event.name,
                "ph": "X",
                "ts": (event.ts - min_ts) * scale,
                "dur": event.dur * scale,
                "pid": event.pid,
                "tid": event.tid,
                "args": {
                    "block": event.block,
                    "warp": event.warp,
                    "scope_id": event.scope_id,
                },
            }
            for event in events
        ],
    }
