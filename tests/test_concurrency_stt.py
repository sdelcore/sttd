"""Concurrency regression tests against the real models.

These exist because of a production failure that no unit test could have caught:
two overlapping transcription requests returned **HTTP 200 with a silently wrong
transcript** ("The Thewnwn the the" in place of the real sentence), and a third
left the CUDA graph permanently poisoned so every later request failed until the
worker was restarted.

The check is against a **golden baseline**, not mutual agreement. Asserting that
N concurrent responses match each other proves nothing when the corruption hits
the whole batch — every response can be identically wrong. Each response is
compared to the transcript the same audio produces when transcribed alone.

Skipped by default — opt in with ``pytest -m integration``. Requires a GPU and
the Parakeet weights.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from voiced.config import TranscriptionConfig
from voiced.transcriber import STT_SAMPLE_RATE, Transcriber, _graphs_are_off

pytestmark = pytest.mark.integration

CONCURRENCY = 8

SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the "
    "morning train departs from platform nine."
)
# Below this the baseline is too thin to reveal corruption, so the fixture
# fails rather than letting a trivial transcript pass every comparison.
MIN_GOLDEN_WORDS = 8


def _synthesize_speech() -> np.ndarray:
    """Render SENTENCE with Kokoro at the STT sample rate.

    Real speech is required. Synthetic tones transcribe to the empty string, and
    an empty baseline makes every comparison below pass regardless of whether
    the race is fixed.
    """
    from scipy import signal as scipy_signal

    from voiced.synthesizer import SAMPLE_RATE as TTS_SAMPLE_RATE
    from voiced.synthesizer import Synthesizer, TTSConfig, check_kokoro_installed

    if not check_kokoro_installed():
        pytest.skip("Kokoro not installed; cannot build a real-speech fixture")

    synth = Synthesizer(TTSConfig(unload_timeout_seconds=0))
    try:
        audio = synth.synthesize(SENTENCE)
    finally:
        # Free the Kokoro VRAM before Parakeet loads; the card is shared.
        synth.shutdown()

    n = int(round(len(audio) * STT_SAMPLE_RATE / TTS_SAMPLE_RATE))
    return scipy_signal.resample(audio, n).astype(np.float32)


@pytest.fixture(scope="module")
def transcriber() -> Transcriber:
    t = Transcriber(TranscriptionConfig(), unload_timeout_minutes=0)
    t.warmup()
    yield t
    t.unload()


@pytest.fixture(scope="module")
def audio(tmp_path_factory) -> np.ndarray:
    speech = _synthesize_speech()
    # Persisted so the negative-control subprocess uses byte-identical audio.
    path = tmp_path_factory.mktemp("concurrency") / "speech.npy"
    np.save(path, speech)
    _speech_path.append(path)
    return speech


_speech_path: list[Path] = []


@pytest.fixture(scope="module")
def golden(transcriber: Transcriber, audio: np.ndarray) -> str:
    """The transcript this audio produces with nothing else running."""
    first = transcriber.transcribe_audio(audio)
    again = transcriber.transcribe_audio(audio)
    assert first == again, (
        "transcription is not deterministic for the fixture audio; "
        "the golden-baseline comparison cannot be trusted"
    )
    # Guard against a vacuous baseline. Comparing "" to "" passes forever.
    assert len(first.split()) >= MIN_GOLDEN_WORDS, (
        f"baseline transcript is too thin to detect corruption: {first!r}. "
        "Every comparison in this module would pass regardless of the fix."
    )
    return first


class TestGraphsDisabled:
    """The CUDA graph decoder must be off. This is the NeMo-upgrade tripwire."""

    def test_graph_decoder_is_disabled_after_load(self, transcriber: Transcriber):
        with transcriber._host.use() as model:
            assert _graphs_are_off(model), (
                "NeMo's CUDA graph decoder is enabled. Concurrent requests will "
                "corrupt transcripts silently."
            )

    def test_decoding_computer_reports_no_graph_mode(self, transcriber: Transcriber):
        with transcriber._host.use() as model:
            computer = model.decoding.decoding.decoding_computer
            assert computer.cuda_graphs_mode is None
            assert computer.allow_cuda_graphs is False


class TestConcurrentTranscription:
    def test_concurrent_results_match_the_golden_baseline(
        self, transcriber: Transcriber, audio: np.ndarray, golden: str
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(transcriber.transcribe_audio, audio) for _ in range(CONCURRENCY)]
            results = [f.result(timeout=300) for f in futures]

        assert len(results) == CONCURRENCY
        mismatches = [r for r in results if r != golden]
        assert not mismatches, (
            f"{len(mismatches)}/{CONCURRENCY} concurrent transcriptions disagreed with "
            f"the single-threaded baseline.\n  golden: {golden!r}\n  got:    {mismatches!r}"
        )

    def test_worker_is_not_poisoned_by_the_storm(
        self, transcriber: Transcriber, audio: np.ndarray, golden: str
    ):
        """The forever-500 check: a sequential request after a storm must work."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(transcriber.transcribe_audio, audio) for _ in range(CONCURRENCY)]
            for f in futures:
                f.result(timeout=300)

        assert transcriber.transcribe_audio(audio) == golden

    def test_mixed_segment_and_plain_calls_agree(
        self, transcriber: Transcriber, audio: np.ndarray, golden: str
    ):
        """timestamps=True and timestamps=False share one decoder; interleave them.

        The two paths reconfigure decoding differently, so overlapping them is a
        sharper probe than repeating one call.
        """
        segmented_golden = " ".join(
            text for _s, _e, text in transcriber.transcribe_audio_with_segments(audio)
        ).strip()
        assert len(segmented_golden.split()) >= MIN_GOLDEN_WORDS, (
            f"segment baseline is too thin: {segmented_golden!r}"
        )

        def plain() -> tuple[str, str]:
            return ("plain", transcriber.transcribe_audio(audio))

        def with_segments() -> tuple[str, str]:
            segments = transcriber.transcribe_audio_with_segments(audio)
            return ("segments", " ".join(text for _s, _e, text in segments).strip())

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [
                pool.submit(plain if i % 2 == 0 else with_segments) for i in range(CONCURRENCY)
            ]
            results = [f.result(timeout=300) for f in futures]

        expected = {"plain": golden, "segments": segmented_golden}
        wrong = [(kind, text) for kind, text in results if text != expected[kind]]
        assert not wrong, (
            f"{len(wrong)}/{CONCURRENCY} interleaved calls diverged from their "
            f"single-threaded baseline: {wrong!r}"
        )


NEGATIVE_CONTROL = textwrap.dedent(
    """
    # Remove the serialization and prove the race comes back. Runs in its own
    # process because it is expected to poison the CUDA context.
    import concurrent.futures, contextlib, sys
    import numpy as np
    import voiced.gpu as gpu

    @contextlib.contextmanager
    def _no_lock(_what):
        yield

    gpu.gpu_exclusive = _no_lock
    import voiced.transcriber as tr
    tr.gpu_exclusive = _no_lock

    from voiced.config import TranscriptionConfig

    audio = np.load(sys.argv[1])

    t = tr.Transcriber(TranscriptionConfig(), unload_timeout_minutes=0)
    t.warmup()
    golden = t.transcribe_audio(audio)
    if len(golden.split()) < 8:
        print(f"BASELINE_TOO_THIN={golden!r}")
        sys.exit(2)

    diverged = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(t.transcribe_audio, audio) for _ in range(8)]
        for f in futures:
            try:
                if f.result(timeout=300) != golden:
                    diverged += 1
            except Exception:
                diverged += 1

    print(f"DIVERGED={diverged}")
    sys.exit(0 if diverged else 1)
    """
)


class TestNegativeControl:
    """A regression test that cannot fail is not a regression test.

    Strips the lock and asserts the corruption returns. If this ever passes
    silently, the positive tests above have stopped proving anything.
    """

    def test_removing_the_lock_reintroduces_the_race(self, audio: np.ndarray):
        assert _speech_path, "the audio fixture did not persist its speech file"
        # pytest's `pythonpath` ini setting does not reach a subprocess, and an
        # installed copy of voiced would shadow the working tree.
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run(
            [sys.executable, "-c", NEGATIVE_CONTROL, str(_speech_path[-1])],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if proc.returncode == 2:
            pytest.fail(f"negative control had a trivial baseline: {proc.stdout.strip()}")
        assert proc.returncode == 0, (
            "Removing gpu_exclusive did NOT reintroduce the race. Either the "
            "hazard moved or this test no longer exercises it — investigate "
            "before trusting the positive tests.\n"
            f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
        )
