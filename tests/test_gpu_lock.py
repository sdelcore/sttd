"""Tests for the process-wide GPU serialization lock.

Pure CPU — no torch, no models. These assert the property that matters: no two
model calls overlap, whatever mix of operations is in flight.
"""

import threading
import time

import pytest

from voiced.gpu import gpu_exclusive, is_held, require_held


class ConcurrencyRecorder:
    """Tracks how many blocks are inside the lock at once."""

    def __init__(self):
        self.active = 0
        self.max_seen = 0
        self.order: list[str] = []
        self._lock = threading.Lock()

    def enter(self, name: str) -> None:
        with self._lock:
            self.active += 1
            self.max_seen = max(self.max_seen, self.active)
            self.order.append(f"+{name}")

    def leave(self, name: str) -> None:
        with self._lock:
            self.active -= 1
            self.order.append(f"-{name}")

    def work(self, name: str, seconds: float = 0.02) -> None:
        """Simulate one model call under the lock."""
        with gpu_exclusive(name):
            self.enter(name)
            time.sleep(seconds)
            self.leave(name)


def run_threads(targets) -> None:
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert all(not t.is_alive() for t in threads), "a thread hung on the GPU lock"


class TestMutualExclusion:
    def test_concurrent_same_op_never_overlaps(self):
        r = ConcurrencyRecorder()
        run_threads([lambda: r.work("stt.transcribe") for _ in range(8)])
        assert r.max_seen == 1

    def test_mixed_ops_never_overlap(self):
        """The historic hazard was cross-model, not intra-STT."""
        r = ConcurrencyRecorder()
        targets = []
        for _ in range(4):
            targets.append(lambda: r.work("stt.transcribe"))
            targets.append(lambda: r.work("tts.chunk"))
            targets.append(lambda: r.work("diar.embed"))
        run_threads(targets)
        assert r.max_seen == 1

    def test_every_block_is_balanced(self):
        r = ConcurrencyRecorder()
        run_threads([lambda: r.work("stt.transcribe") for _ in range(6)])
        assert r.active == 0
        # Entries and exits must strictly alternate if nothing overlapped.
        assert all(
            a.startswith("+") and b.startswith("-")
            for a, b in zip(r.order[::2], r.order[1::2], strict=True)
        )

    def test_lock_is_released_when_the_body_raises(self):
        with pytest.raises(ValueError):
            with gpu_exclusive("boom"):
                raise ValueError("boom")
        r = ConcurrencyRecorder()
        run_threads([lambda: r.work("stt.transcribe") for _ in range(3)])
        assert r.max_seen == 1

    def test_reentrant_within_one_thread(self):
        """A wrapped helper calling another wrapped helper must not deadlock."""
        with gpu_exclusive("outer"):
            with gpu_exclusive("inner"):
                assert is_held()


class TestHeldIntrospection:
    def test_not_held_by_default(self):
        assert is_held() is False

    def test_held_inside_the_block(self):
        with gpu_exclusive("stt.transcribe"):
            assert is_held() is True
        assert is_held() is False

    def test_not_held_when_another_thread_holds_it(self):
        entered = threading.Event()
        release = threading.Event()
        seen = []

        def holder():
            with gpu_exclusive("holder"):
                entered.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        assert entered.wait(timeout=5)
        seen.append(is_held())
        release.set()
        t.join(timeout=5)

        assert seen == [False]


class TestRequireHeld:
    def test_raises_when_not_held(self):
        with pytest.raises(RuntimeError, match="without holding the GPU lock"):
            require_held("stt.transcribe")

    def test_passes_when_held(self):
        with gpu_exclusive("stt.transcribe"):
            require_held("stt.transcribe")

    def test_message_names_the_caller_and_the_fix(self):
        with pytest.raises(RuntimeError) as excinfo:
            require_held("diar.embed")
        message = str(excinfo.value)
        assert "diar.embed" in message
        assert "gpu_exclusive" in message
