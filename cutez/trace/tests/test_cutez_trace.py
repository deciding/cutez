import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, '.')
from trace.format import (
    ChromeTraceEvent,
    PackedEvent,
    decode_packed_event,
    decode_ring_events,
    pair_complete_events,
    trace_json_payload,
)
from trace.examples import run_sample_trace
from trace.session import CutezTraceSession


def pack_event(clock_lo: int, clock_hi_low11: int, scope_id: int, is_start: bool) -> int:
    hi = (clock_hi_low11 & 0x7FF) | ((scope_id & 0xFF) << 23)
    if not is_start:
        hi |= 1 << 31
    return ((hi & 0xFFFFFFFF) << 32) | (clock_lo & 0xFFFFFFFF)


def to_signed_int64(word: int) -> int:
    return word if word < (1 << 63) else word - (1 << 64)


def test_decode_packed_event_extracts_clock_scope_and_direction():
    word = pack_event(clock_lo=17, clock_hi_low11=9, scope_id=5, is_start=True)

    event = decode_packed_event(word, block=0, warp=1, logical_index=0)

    assert event.clock_lo == 17
    assert event.clock_hi_low11 == 9
    assert event.scope_id == 5
    assert event.is_start is True


def test_decode_ring_events_reconstructs_latest_entries_after_wrap():
    words = [
        pack_event(10, 0, 1, True),
        pack_event(11, 0, 1, False),
        pack_event(12, 0, 2, True),
        pack_event(13, 0, 2, False),
    ]
    wrapped_words = [
        pack_event(102, 0, 3, True),
        pack_event(103, 0, 3, False),
        words[2],
        words[3],
    ]

    events = decode_ring_events(
        wrapped_words,
        block=0,
        warp=0,
        total_event_count=6,
    )

    assert [event.clock_lo for event in events] == [12, 13, 102, 103]
    assert [event.logical_index for event in events] == [2, 3, 4, 5]


def test_decode_ring_events_reconstructs_latest_entries_for_non_power_of_two_capacity():
    wrapped_words = [
        pack_event(202, 0, 3, True),
        pack_event(203, 0, 3, False),
        pack_event(102, 0, 2, False),
    ]

    events = decode_ring_events(
        wrapped_words,
        block=1,
        warp=0,
        total_event_count=5,
    )

    assert [event.clock_lo for event in events] == [102, 202, 203]
    assert [event.logical_index for event in events] == [2, 3, 4]


def test_pair_complete_events_builds_positive_duration_chrome_events():
    words = [
        pack_event(100, 0, 7, True),
        pack_event(112, 0, 7, False),
    ]
    decoded = decode_ring_events(words, block=2, warp=1, total_event_count=2)

    paired = pair_complete_events(decoded, region_names={7: "mma"})

    assert len(paired) == 1
    assert paired[0].name == "mma"
    assert paired[0].dur > 0
    assert paired[0].block == 2
    assert paired[0].warp == 1


def test_pair_complete_events_preserves_zero_duration_chrome_events():
    words = [
        pack_event(100, 0, 8, True),
        pack_event(100, 0, 8, False),
    ]
    decoded = decode_ring_events(words, block=0, warp=0, total_event_count=2)

    paired = pair_complete_events(decoded, region_names={8: "sync"})

    assert len(paired) == 1
    assert paired[0].name == "sync"
    assert paired[0].dur == 0


def test_trace_json_payload_emits_chrome_trace_shape():
    event = ChromeTraceEvent(
        name="load",
        ts=0,
        dur=5,
        pid=0,
        tid=32,
        block=0,
        warp=1,
        scope_id=3,
    )

    payload = trace_json_payload([event])

    assert payload["displayTimeUnit"] == "ns"
    assert payload["traceEvents"][0]["ph"] == "X"
    assert payload["traceEvents"][0]["name"] == "load"


def test_public_package_exports_include_session_and_runner():
    import trace

    assert hasattr(trace, "CutezTraceSession")
    assert hasattr(trace, "trace_json_payload")
    assert hasattr(trace, "run_sample_trace")
    assert hasattr(trace, "init_clock")
    assert hasattr(trace, "clock_record")
    assert hasattr(trace, "finanlize_clock")
    assert hasattr(trace, "SharedStorage")


def test_package_root_defers_examples_import_until_runner_access(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        check=True,
    )
    python_bin = venv_dir / "bin" / "python"
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "--no-deps", "-e", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            str(python_bin),
            "-c",
            (
                "import sys; "
                "import trace; "
                "assert 'trace.examples' not in sys.modules; "
                "_ = trace.run_sample_trace; "
                "assert 'trace.examples' in sys.modules"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_installed_environment_can_import_cutez_trace_examples(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    venv_dir = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        check=True,
    )
    python_bin = venv_dir / "bin" / "python"
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "--no-deps", "-e", str(repo_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(python_bin), "-c", "import trace.examples; import trace.core"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_session_allocates_one_segment_per_block_warp():
    session = CutezTraceSession(blocks=2, warps_per_block=4, segment_bytes=32)

    assert session.segment_words == 4
    assert session.total_segments == 8
    assert session.buffer_numel == 32


def test_session_decodes_flat_buffer_by_block_and_warp():
    session = CutezTraceSession(blocks=1, warps_per_block=2, segment_bytes=32)
    words = torch.tensor(
        [
            pack_event(10, 0, 1, True),
            to_signed_int64(pack_event(12, 0, 1, False)),
            0,
            0,
            pack_event(20, 0, 2, True),
            to_signed_int64(pack_event(23, 0, 2, False)),
            0,
            0,
        ],
        dtype=torch.int64,
    )

    decoded = session.decode_buffer(words, counts={(0, 0): 2, (0, 1): 2})

    assert len(decoded[(0, 0)]) == 2
    assert len(decoded[(0, 1)]) == 2
    assert decoded[(0, 1)][0].warp == 1


def test_session_write_trace_json_creates_file(tmp_path: Path):
    session = CutezTraceSession(
        blocks=1,
        warps_per_block=1,
        segment_bytes=32,
        sm_clock_khz=500_000,
    )
    words = torch.tensor(
        [
            pack_event(100, 0, 9, True),
            to_signed_int64(pack_event(106, 0, 9, False)),
            0,
            0,
        ],
        dtype=torch.int64,
    )
    path = tmp_path / "trace.json"

    session.write_trace_json(
        path,
        words,
        counts={(0, 0): 2},
        region_names={9: "epilogue"},
    )

    payload = json.loads(path.read_text())
    assert payload["displayTimeUnit"] == "ns"
    assert payload["traceEvents"][0]["name"] == "epilogue"
    assert payload["traceEvents"][0]["ts"] == 0
    assert payload["traceEvents"][0]["dur"] == 12
    assert payload["traceEvents"][0]["args"]["warp"] == 0
    assert payload["traceEvents"][0]["args"]["clock_rate_khz"] == 500_000


def test_session_decode_buffer_rejects_invalid_buffer_length():
    session = CutezTraceSession(blocks=1, warps_per_block=1, segment_bytes=32)
    words = torch.zeros(session.buffer_numel - 1, dtype=torch.int64)

    with pytest.raises(ValueError, match="buffer_numel"):
        session.decode_buffer(words, counts={(0, 0): 0})


def test_session_decode_buffer_rejects_invalid_buffer_dtype():
    session = CutezTraceSession(blocks=1, warps_per_block=1, segment_bytes=32)
    words = torch.zeros(session.buffer_numel, dtype=torch.int32)

    with pytest.raises(TypeError, match="torch.int64"):
        session.decode_buffer(words, counts={(0, 0): 0})


@pytest.mark.cuda
def test_run_sample_trace_writes_nonempty_trace_json(tmp_path: Path):
    trace_path = tmp_path / "sample_trace.json"

    result = run_sample_trace(trace_path=trace_path)

    assert trace_path.exists()
    payload = json.loads(trace_path.read_text())
    complete = [event for event in payload["traceEvents"] if event.get("ph") == "X"]
    assert payload["displayTimeUnit"] == "ns"
    assert complete
    assert {event["name"] for event in complete}
    assert all(event["dur"] >= 0 for event in complete)
    assert result["counts"]


@pytest.mark.cuda
def test_run_sample_trace_rejects_unsupported_warp_selection(tmp_path: Path):
    trace_path = tmp_path / "unsupported_trace.json"

    with pytest.raises(ValueError, match=r"active_warps=\(0,\)"):
        run_sample_trace(trace_path=trace_path, active_warps=(1,))


@pytest.mark.cuda
def test_run_sample_trace_supports_two_selected_warps(tmp_path: Path):
    trace_path = tmp_path / "two_warp_trace.json"

    run_sample_trace(trace_path=trace_path, active_warps=(0, 1), iters=1)

    payload = json.loads(trace_path.read_text())
    complete = [event for event in payload["traceEvents"] if event.get("ph") == "X"]
    assert {event["args"]["warp"] for event in complete} == {0, 1}


@pytest.mark.cuda
def test_run_sample_trace_retains_latest_events_after_wrap(tmp_path: Path):
    trace_path = tmp_path / "wrapped_trace.json"

    result = run_sample_trace(trace_path=trace_path, active_warps=(0,), iters=8)

    payload = json.loads(trace_path.read_text())
    complete = [event for event in payload["traceEvents"] if event.get("ph") == "X"]
    assert complete
    assert result["counts"][(0, 0)] > result["session"].segment_words


@pytest.mark.cuda
def test_run_sample_trace_does_not_record_unselected_warp(tmp_path: Path):
    trace_path = tmp_path / "warp_zero_only_trace.json"

    result = run_sample_trace(trace_path=trace_path, active_warps=(0,), iters=1)

    segment_words = result["session"].segment_words
    warp_one_segment = result["buffer"][segment_words : 2 * segment_words]
    assert torch.count_nonzero(warp_one_segment).item() == 0
