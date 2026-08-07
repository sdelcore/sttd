"""Process-wide serialization of GPU inference.

## The invariant

Every call into a loaded model object runs inside ``gpu_exclusive()``.

Two independent hazards make this necessary, and they are not backups for each
other:

1. **NeMo is not re-entrant.** ``_transcribe_on_begin`` freezes the encoder,
   decoder and joint, and stashes ``training``, ``dither`` and ``pad_to`` on the
   module; ``_transcribe_on_end`` restores them and deletes the saved grad map.
   Two overlapping ``transcribe()`` calls on one model tear down each other's
   state. This is plain attribute mutation on an ``nn.Module`` — it races on CPU
   and it races with CUDA graphs disabled. Only mutual exclusion fixes it.
   Upstream has no lock of its own: NVIDIA/NeMo#15771.

2. **CUDA graph capture is process-scoped.** "Within a process, only one capture
   may be underway at a time", and no non-captured CUDA work may run in the
   process on any thread while capture is underway. A lock cannot make that safe
   for work it does not cover, so voiced disables the graph decoder outright in
   ``transcriber._load_model``. That is a correctness requirement, not a tuning
   knob — see the check in that function.

The second hazard also explains the nastiest symptom this fixed: NeMo captures
with ``capture_error_mode="thread_local"``, which switches CUDA's cross-thread
check off, so a bystander thread got undefined behaviour instead of an error.
That surfaced as a silently wrong transcript returned with HTTP 200.

## Lock ordering

``gpu_exclusive()`` is the innermost lock. Acquire it last and release it first.
Holding ``ModelHost._lock`` and then taking this lock is correct and happens on
the unload path; the reverse order is forbidden and would deadlock.

The lock is re-entrant so that a wrapper which already holds it may call another
wrapped helper without deadlocking.
"""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Re-entrant so nested wrapped calls in one thread do not deadlock.
_GPU_LOCK = threading.RLock()

# Contention above this is worth knowing about; it means a request queued.
SLOW_WAIT_SECONDS = 0.1
# A hold this long is a stall, not a transcription. Kokoro runs at RTF ~0.05 and
# Parakeet decodes 60s of audio in under 200ms, so nothing legitimate is close.
SLOW_HOLD_SECONDS = 5.0


@contextmanager
def gpu_exclusive(what: str) -> Iterator[None]:
    """Run a GPU model call with exclusive access to the device.

    ``what`` names the operation and appears in contention warnings, so pass
    something greppable like ``"stt.transcribe"``.
    """
    requested = time.monotonic()
    _GPU_LOCK.acquire()
    waited = time.monotonic() - requested
    if waited > SLOW_WAIT_SECONDS:
        logger.warning(f"gpu: {what} waited {waited * 1000:.0f}ms for the device")

    held_from = time.monotonic()
    try:
        yield
    finally:
        held = time.monotonic() - held_from
        _GPU_LOCK.release()
        if held > SLOW_HOLD_SECONDS:
            logger.warning(f"gpu: {what} held the device for {held:.1f}s")


def is_held() -> bool:
    """Whether the calling thread currently holds the GPU lock."""
    if _GPU_LOCK.acquire(blocking=False):
        try:
            # Re-acquiring a lock this thread already owns lands at depth > 1;
            # a fresh acquire of a free lock lands at depth 1.
            return _GPU_LOCK._recursion_count() > 1  # type: ignore[attr-defined]
        finally:
            _GPU_LOCK.release()
    return False


def require_held(what: str) -> None:
    """Raise unless the calling thread holds the GPU lock.

    For helpers that touch a model but do not take the lock themselves. A
    forgotten lock is exactly the failure that returned a silently wrong
    transcript, so this raises rather than warning.
    """
    if not is_held():
        raise RuntimeError(
            f"{what} touched a GPU model without holding the GPU lock. "
            "Wrap the call in voiced.gpu.gpu_exclusive() — see voiced/gpu.py."
        )
