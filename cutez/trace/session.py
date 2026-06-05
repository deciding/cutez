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
from cutlass.cute.runtime import from_dlpack

from .core import TraceConfig, get_smem_cap
from .format import (
    ChromeTraceEvent,
    decode_ring_events,
    pair_complete_events,
    trace_json_payload,
)
from .core import get_region_names


@dataclass
class CutezTraceSession:
    warps_per_block: int
    trace_path: str | Path
    sm_smem_available_bytes: int
    total_blocks: int = 2
    blocks_per_sm: int = 1
    region_names: dict[int, str] | None = None
    device: str | torch.device = "cuda"
    sm_clock_khz: int | None = None
    enabled: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.trace_path = Path(self.trace_path)
        smem_cap = get_smem_cap()
        self.block_smem_bytes = self.sm_smem_available_bytes // self.blocks_per_sm
        self.segment_bytes = self.block_smem_bytes // self.warps_per_block
        if self.segment_bytes % 8 != 0:
            raise ValueError("derived segment_bytes must be divisible by 8")
        self.segment_words = self.segment_bytes // 8
        self.block_smem_words = self.block_smem_bytes // 8
        self.total_segments = self.total_blocks * self.warps_per_block
        self.buffer_numel = self.block_smem_words * self.total_blocks
        self.trace_config = TraceConfig(
            block_smem_bytes=self.block_smem_bytes,
            segment_bytes=self.segment_bytes,
            smem_words=self.block_smem_words,
            enabled=self.enabled,
            smem_capacity_bytes=smem_cap,
            total_blocks=self.total_blocks,
            warps_per_block=self.warps_per_block,
            sm_smem_available_bytes=self.sm_smem_available_bytes,
            verbose=self.verbose,
        )
        if not self.enabled:
            self.buffer_tensor = None
            self.buffer = None
        else:
            self.buffer_tensor = torch.zeros(
                self.buffer_numel, dtype=torch.int64, device=self.device
            )
            self.buffer = from_dlpack(self.buffer_tensor, assumed_align=8)

    def reset_buffer(self):
        if self.buffer_tensor is not None:
            self.buffer_tensor.zero_()

    def resolve_sm_clock_khz(self) -> int:
        if self.sm_clock_khz is not None:
            return self.sm_clock_khz
        props = torch.cuda.get_device_properties(self.device)
        return int(props.clock_rate)

    def ticks_to_ns(self, ticks: int, *, sm_clock_khz: int | None = None) -> int:
        # `clock_rate` is reported in kHz, so one tick is 1e6 / kHz nanoseconds.
        clock_khz = (
            sm_clock_khz if sm_clock_khz is not None else self.resolve_sm_clock_khz()
        )
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
        self, words: torch.Tensor | None = None
    ) -> dict[tuple[int, int], list]:
        """Decode the flat output buffer into per-warp event lists.

        The caller may pass a flat `torch.int64` buffer explicitly; otherwise
        the session decodes its owned `buffer_tensor`. Each warp-owned segment is
        decoded by filtering zero-valued unused slots and ordering the retained
        nonzero events chronologically by reconstructed clock value.
        """
        if words is None:
            if self.buffer_tensor is None:
                return {}
            words = self.buffer_tensor
        if words.dtype != torch.int64:
            raise TypeError(
                f"decode_buffer expects a torch.int64 buffer, got {words.dtype}"
            )
        if words.numel() != self.buffer_numel:
            raise ValueError(
                f"decode_buffer expected buffer_numel={self.buffer_numel}, got {words.numel()}"
            )

        flat = words.detach().cpu().reshape(-1).tolist()
        out = {}
        for block in range(self.total_blocks):
            for warp in range(self.warps_per_block):
                segment = block * self.warps_per_block + warp
                start = segment * self.segment_words
                stop = start + self.segment_words
                out[(block, warp)] = decode_ring_events(
                    flat[start:stop],
                    block=block,
                    warp=warp,
                )
        return out

    def write_trace_json(self, max_blocks: int | None = None) -> Path:
        """Decode a trace buffer and write a Chrome trace JSON file.

        This method decodes the session-owned flat buffer, pairs
        begin/end events, converts `%clock` ticks to approximate nanoseconds
        using the device SM clock rate, and returns the written path.

        When *max_blocks* is provided, only blocks 0 .. max_blocks-1 are
        included in the output. The starting timestamp is normalized to
        the minimum among the selected blocks.
        """
        if self.buffer_tensor is None:
            return self.trace_path
        decoded = self.decode_buffer(self.buffer_tensor)
        region_names = (
            self.region_names if self.region_names is not None else get_region_names()
        )
        paired = []
        for (block, warp), events in decoded.items():
            if max_blocks is not None and block >= max_blocks:
                continue
            paired.extend(pair_complete_events(events, region_names=region_names))
        sm_clock_khz = self.resolve_sm_clock_khz()
        payload = trace_json_payload(
            self._events_to_chrome_time(paired, sm_clock_khz=sm_clock_khz)
        )
        for event in payload["traceEvents"]:
            event["args"]["clock_rate_khz"] = sm_clock_khz
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(json.dumps(payload))
        return self.trace_path

    def debug_dump_segments(self, words: torch.Tensor | None = None) -> dict:
        """Return raw retained-clock and paired-event diagnostics per block/warp.

        This is a host-side debugging helper for investigating decode issues before
        trace JSON normalization.
        """
        decoded = self.decode_buffer(words)
        debug = {}
        for block in range(self.total_blocks):
            for warp in range(self.warps_per_block):
                events = decoded[(block, warp)]
                paired = pair_complete_events(events, region_names=self.region_names)
                debug[(block, warp)] = {
                    "raw_clocks": [event.raw_clock for event in events],
                    "scopes": [event.scope_id for event in events],
                    "is_start": [event.is_start for event in events],
                    "paired_starts": [event.ts for event in paired],
                    "paired_durations": [event.dur for event in paired],
                }
        return debug
