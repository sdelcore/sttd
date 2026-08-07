"""Tests for the worker liveness deadline and fatal-CUDA-error classification.

GPU work is serialized process-wide, so a single wedged call would otherwise
hold the device and stall every later request forever. These cover the backstop
that turns that into a loud, bounded failure.
"""

import os
import time

import numpy as np
import pytest

from voiced import worker
from voiced.config import Config
from voiced.worker_host import (
    WorkerHost,
    WorkerTimeoutError,
)

# ----- fake backends (module level so spawn can pickle them by reference) -----


class HangingTranscriber:
    device = "cpu"

    def warmup(self):
        pass

    def transcribe_audio(self, audio, sample_rate):
        time.sleep(3600)

    def transcribe_file(self, path):
        time.sleep(3600)

    def transcribe_audio_with_segments(self, audio, sample_rate):
        return [(0.0, 1.0, "ok")]

    def transcribe_partial(self, audio):
        return "partial"


class QuickTranscriber(HangingTranscriber):
    def transcribe_audio(self, audio, sample_rate):
        return f"text:{len(audio)}"


def hanging_factory(config):
    return HangingTranscriber()


def quick_factory(config):
    return QuickTranscriber()


def make_host(factory, **kwargs) -> WorkerHost:
    return WorkerHost(
        Config(),
        transcriber_factory=factory,
        start_timeout=60.0,
        **kwargs,
    )


class TestDeadlineComputation:
    """Pure arithmetic — no worker spawned."""

    def test_array_op_scales_with_audio_duration(self):
        host = make_host(quick_factory)
        audio = np.zeros(16000 * 60, dtype=np.float32)  # 60 seconds
        deadline = host._deadline_for(
            "stt.transcribe_audio", {"audio": audio, "sample_rate": 16000}
        )
        assert deadline == pytest.approx(600.0)

    def test_array_op_has_a_floor_for_short_audio(self):
        host = make_host(quick_factory)
        audio = np.zeros(1600, dtype=np.float32)  # 0.1 seconds
        deadline = host._deadline_for(
            "stt.transcribe_audio", {"audio": audio, "sample_rate": 16000}
        )
        assert deadline == pytest.approx(30.0)

    def test_path_op_uses_the_fixed_ceiling(self):
        """The parent cannot know a duration for a path — worker.py sends the
        path, not audio — so path ops get a generous fixed deadline."""
        host = make_host(quick_factory)
        deadline = host._deadline_for("stt.transcribe_file", {"path": "/some/long.wav"})
        assert deadline == pytest.approx(900.0)

    def test_missing_audio_falls_back_to_the_floor(self):
        host = make_host(quick_factory)
        assert host._deadline_for("stt.transcribe_partial", {}) == pytest.approx(30.0)

    def test_deadlines_are_configurable(self):
        host = make_host(quick_factory, array_op_min_deadline=5.0, path_op_deadline=50.0)
        assert host._deadline_for("stt.transcribe_partial", {}) == pytest.approx(5.0)
        assert host._deadline_for("stt.transcribe_file", {"p": 1}) == pytest.approx(50.0)


class TestWedgedWorker:
    def test_hung_op_raises_worker_timeout(self):
        host = make_host(hanging_factory, array_op_min_deadline=2.0)
        try:
            with pytest.raises(WorkerTimeoutError):
                host.request(
                    "stt.transcribe_audio",
                    audio=np.zeros(1600, dtype=np.float32),
                    sample_rate=16000,
                )
        finally:
            host.shutdown()

    def test_hung_worker_is_killed_not_left_running(self):
        host = make_host(hanging_factory, array_op_min_deadline=2.0)
        try:
            with host._host.use() as handle:
                pid = handle.process.pid
            with pytest.raises(WorkerTimeoutError):
                host.request(
                    "stt.transcribe_audio",
                    audio=np.zeros(1600, dtype=np.float32),
                    sample_rate=16000,
                )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("the wedged worker was left running")
        finally:
            host.shutdown()

    def test_a_hung_op_is_not_retried(self):
        """Retrying a hang just hangs again; it must surface immediately."""
        host = make_host(hanging_factory, array_op_min_deadline=2.0)
        try:
            started = time.monotonic()
            with pytest.raises(WorkerTimeoutError):
                host.request(
                    "stt.transcribe_audio",
                    audio=np.zeros(1600, dtype=np.float32),
                    sample_rate=16000,
                )
            # One deadline, not two. Allow generous slack for spawn and kill.
            assert time.monotonic() - started < 45
        finally:
            host.shutdown()

    def test_normal_ops_are_unaffected(self):
        host = make_host(quick_factory, array_op_min_deadline=30.0)
        try:
            out = host.request(
                "stt.transcribe_audio",
                audio=np.zeros(1600, dtype=np.float32),
                sample_rate=16000,
            )
            # STT results are wrapped so the parent can report the device
            # without a CUDA-touching import (see worker.py's protocol notes).
            assert out == {"value": "text:1600", "device": "cpu"}
        finally:
            host.shutdown()


class TestFatalCudaClassification:
    """Classified by exception TYPE so an upstream reword cannot silently turn a
    fatal condition into an infinite retry."""

    def test_context_fault_is_fatal(self):
        exc = RuntimeError("CUDA error: operation not permitted when stream is capturing")
        assert worker._is_unrecoverable_cuda_error(exc) is True

    def test_oom_is_not_fatal(self):
        import torch

        exc = torch.cuda.OutOfMemoryError("CUDA out of memory")
        assert worker._is_unrecoverable_cuda_error(exc) is False

    def test_accelerator_error_is_fatal(self):
        import torch

        accelerator_error = getattr(torch, "AcceleratorError", None)
        if accelerator_error is None:
            pytest.skip("this torch has no AcceleratorError")
        assert worker._is_unrecoverable_cuda_error(accelerator_error("CUDA error: bad")) is True

    def test_ordinary_errors_are_not_fatal(self):
        assert worker._is_unrecoverable_cuda_error(ValueError("bad audio")) is False
        assert worker._is_unrecoverable_cuda_error(FileNotFoundError("nope")) is False

    def test_a_plain_runtime_error_is_not_fatal(self):
        """Only CUDA-tagged RuntimeErrors kill the worker."""
        assert (
            worker._is_unrecoverable_cuda_error(RuntimeError("Parakeet returned no results"))
            is False
        )
