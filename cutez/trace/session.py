"""Host-side buffer management for SMEM-flushed cutez traces.

`CutezTraceSession` allocates the GMEM output tensor shape expected by the
example kernels and provides decode + JSON writing helpers.

Chrome trace viewers require real time values. This session converts raw GPU
``%clock`` ticks into nanoseconds using the device SM clock rate reported by
CUDA. That conversion is approximate and does not provide QuACK-style runtime
calibration across SMs, but it is a materially more honest time base than
exporting raw ticks as if they were wall-clock units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .format import ChromeTraceEvent, decode_ring_events, pair_complete_events, trace_json_payload


@dataclass
class CutezTraceSession:
    blocks: int
    warps_per_block: int
    segment_bytes: int
    device: str | torch.device = "cuda"
    sm_clock_khz: int | None = None

    def __post_init__(self):
        if self.segment_bytes % 8 != 0:
            raise ValueError("segment_bytes must be divisible by 8")
        self.segment_words = self.segment_bytes // 8
        self.total_segments = self.blocks * self.warps_per_block
        self.buffer_numel = self.total_segments * self.segment_words

    def allocate_buffer(self) -> torch.Tensor:
        """Allocate the flat GMEM output buffer expected by the example kernels.

        The returned `torch.int64` tensor has one contiguous segment per
        `(block, wid)` pair, where the warp id is also the segment id used by
        `my_trace.init_clock(...)` and `my_trace.finanlize_clock(...)`.
        """
        return torch.zeros(self.buffer_numel, dtype=torch.int64, device=self.device)

    def resolve_sm_clock_khz(self) -> int:
        if self.sm_clock_khz is not None:
            return self.sm_clock_khz
        props = torch.cuda.get_device_properties(self.device)
        return int(props.clock_rate)

    def ticks_to_ns(self, ticks: int, *, sm_clock_khz: int | None = None) -> int:
        # `clock_rate` is reported in kHz, so one tick is 1e6 / kHz nanoseconds.
        clock_khz = sm_clock_khz if sm_clock_khz is not None else self.resolve_sm_clock_khz()
        return int(round((ticks * 1_000_000.0) / clock_khz))

    def _events_to_chrome_time(
        self, events: list[ChromeTraceEvent], *, sm_clock_khz: int
    ) -> list[ChromeTraceEvent]:
        return [
            ChromeTraceEvent(
                name=event.name,
                ts=self.ticks_to_ns(event.ts, sm_clock_khz=sm_clock_khz),
                dur=self.ticks_to_ns(event.dur, sm_clock_khz=sm_clock_khz),
                pid=event.pid,
                tid=event.tid,
                block=event.block,
                warp=event.warp,
                scope_id=event.scope_id,
            )
            for event in events
        ]

    def decode_buffer(
        self, words: torch.Tensor, *, counts: dict[tuple[int, int], int]
    ) -> dict[tuple[int, int], list]:
        """Decode the flat output buffer into per-warp event lists.

        The caller passes the flat `torch.int64` buffer and `counts[(block,
        warp)]`, where each count is the total `clock_idx` value produced for
        that warp. The return value maps `(block, warp)` to the retained
        `PackedEvent` list reconstructed from that warp's segment.
        """
        if words.dtype != torch.int64:
            raise TypeError(f"decode_buffer expects a torch.int64 buffer, got {words.dtype}")
        if words.numel() != self.buffer_numel:
            raise ValueError(
                f"decode_buffer expected buffer_numel={self.buffer_numel}, got {words.numel()}"
            )

        flat = words.detach().cpu().reshape(-1).tolist()
        out = {}
        for block in range(self.blocks):
            for warp in range(self.warps_per_block):
                segment = block * self.warps_per_block + warp
                start = segment * self.segment_words
                stop = start + self.segment_words
                out[(block, warp)] = decode_ring_events(
                    flat[start:stop],
                    block=block,
                    warp=warp,
                    total_event_count=counts.get((block, warp), 0),
                )
        return out

    def write_trace_json(
        self,
        path: str | Path,
        words: torch.Tensor,
        *,
        counts: dict[tuple[int, int], int],
        region_names: dict[int, str] | None = None,
    ) -> Path:
        """Decode a trace buffer and write a Chrome trace JSON file.

        The caller provides the raw buffer plus per-warp `clock_idx` totals.
        This method decodes each `(block, wid)` segment, pairs begin/end events,
        converts `%clock` ticks to approximate nanoseconds using the device SM
        clock rate, and returns the written path.
        """
        decoded = self.decode_buffer(words, counts=counts)
        paired = []
        for events in decoded.values():
            paired.extend(pair_complete_events(events, region_names=region_names))
        sm_clock_khz = self.resolve_sm_clock_khz()
        payload = trace_json_payload(self._events_to_chrome_time(paired, sm_clock_khz=sm_clock_khz))
        for event in payload["traceEvents"]:
            event["args"]["clock_rate_khz"] = sm_clock_khz
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path
