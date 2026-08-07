"""Transcription engine using NVIDIA Parakeet-TDT (NeMo)."""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from voiced.audio_codec import normalize_audio
from voiced.config import TranscriptionConfig
from voiced.device import resolve_device_config
from voiced.gpu import gpu_exclusive
from voiced.model_host import ModelHost

logger = logging.getLogger(__name__)

STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
STT_SAMPLE_RATE = 16000


def _empty_cuda_cache(_model: Any) -> None:
    import torch

    if torch.cuda.is_available():
        with gpu_exclusive("stt.unload"):
            torch.cuda.empty_cache()


class CudaGraphsStillEnabledError(RuntimeError):
    """NeMo kept its CUDA graph decoder after we asked for the graph-free path."""


def _disable_cuda_graph_decoder(model: Any) -> None:
    """Pin NeMo's TDT decoder to the graph-free ``torch_impl`` path.

    CUDA graph capture is process-scoped: while a capture is underway no other
    thread in the process may issue CUDA work. voiced serializes model calls
    (see voiced/gpu.py), but that cannot cover CUDA work outside those calls —
    the cyclic GC frees tensors on whichever thread trips the allocation
    threshold, and tracebacks pin tensors too. So the capture is removed rather
    than guarded.

    Removing it also removes the ``LabelLoopingState`` buffers, which live only
    on the graph path. Those buffers produced the silently wrong transcript that
    a concurrent request returned with HTTP 200.

    Raises on failure. A daemon that starts with graphs live looks healthy and
    corrupts transcripts under load, which is precisely the failure this is here
    to prevent.
    """
    from omegaconf import open_dict

    with open_dict(model.cfg.decoding):
        # Under `greedy`, not `greedy_batch` — change_decoding_strategy writes
        # cfg.decoding back, so the flag has to live where it will be re-read.
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding, verbose=False)

    if _graphs_are_off(model):
        return

    # NeMo mutates use_cuda_graph_decoder independently in its own fallback
    # path, so the config flag is not evidence. Escalate to the explicit API.
    logger.error("CUDA graph decoder survived the config change; forcing it off")
    try:
        from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs

        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(model, "decoding.decoding")
    except Exception as e:
        raise CudaGraphsStillEnabledError(
            f"Could not disable NeMo's CUDA graph decoder: {e}. Refusing to start — "
            "with graphs live, concurrent requests corrupt transcripts silently."
        ) from e

    if not _graphs_are_off(model):
        raise CudaGraphsStillEnabledError(
            "NeMo's CUDA graph decoder is still enabled after both the config change "
            "and disable_cuda_graphs_recursive(). Refusing to start — with graphs live, "
            "concurrent requests corrupt transcripts silently. A NeMo upgrade has "
            "probably moved the flag; see voiced/gpu.py for why this matters."
        )


def _graphs_are_off(model: Any) -> bool:
    """Read graph state off the decoding computer, not off the config flag."""
    try:
        computer = model.decoding.decoding.decoding_computer
    except AttributeError:
        # No computer means no label-looping decoder, hence no capture.
        return True
    mode = getattr(computer, "cuda_graphs_mode", None)
    allow = getattr(computer, "allow_cuda_graphs", False)
    return mode is None and not allow


class Transcriber:
    """Wrapper around NeMo Parakeet-TDT for speech-to-text transcription."""

    def __init__(
        self,
        config: TranscriptionConfig | None = None,
        *,
        unload_timeout_minutes: int = 15,
    ):
        """Initialize the transcriber.

        Args:
            config: Transcription configuration. Uses defaults if not provided.
            unload_timeout_minutes: Auto-unload the model after this many minutes
                idle (0 = never). Shared with TTS via the app config.
        """
        self.config = config or TranscriptionConfig()
        self._replacements = [
            (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), right)
            for wrong, right in self.config.replacements.items()
        ]
        self._device: str | None = None
        self._host: ModelHost[Any] = ModelHost(
            loader=self._load_model,
            idle_timeout=(unload_timeout_minutes * 60 if unload_timeout_minutes > 0 else None),
            on_unload=_empty_cuda_cache,
            name=f"parakeet({STT_MODEL})",
        )

    @property
    def device(self) -> str:
        """The device the model runs on. Resolved without loading if not yet loaded."""
        return self._device or resolve_device_config(self.config.device).device

    def warmup(self) -> None:
        """Load the model and run one decode, so the first real request is not
        the one paying for lazy initialisation."""
        silence = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        with self._host.use() as model:
            with gpu_exclusive("stt.warmup"):
                model.transcribe([silence], timestamps=False, verbose=False)

    def _load_model(self) -> Any:
        """Load the Parakeet ASR model from HuggingFace via NeMo."""
        import nemo.collections.asr as nemo_asr

        device = resolve_device_config(self.config.device).device
        self._device = device

        logger.info(f"Loading model '{STT_MODEL}' on {device}")

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=STT_MODEL)
        model = model.to(device)
        model.eval()
        _disable_cuda_graph_decoder(model)
        logger.info("CUDA graph decoder disabled; using the graph-free decode path")
        return model

    def _run(self, source: Any, *, timestamps: bool = False) -> Any:
        """Run the model and return the first result.

        The lock is taken inside the lease, around the model call only, so a
        first request that has to load Parakeet does not block TTS for the
        ~30s load.
        """
        with self._host.use() as model:
            with gpu_exclusive("stt.transcribe"):
                results = model.transcribe([source], timestamps=timestamps)
        if not results:
            raise RuntimeError("Parakeet returned no results")
        return results[0]

    def _fix(self, text: str) -> str:
        """Apply configured word replacements to transcribed text."""
        for pattern, replacement in self._replacements:
            text = pattern.sub(replacement, text)
        return text

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe an audio file."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing file: {audio_path}")
        result = self._run(str(audio_path))
        return self._fix((result.text or "").strip())

    def transcribe_file_with_segments(
        self, audio_path: str | Path
    ) -> list[tuple[float, float, str]]:
        """Transcribe an audio file and return segments with timestamps."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing file with segments: {audio_path}")
        result = self._run(str(audio_path), timestamps=True)
        return self._fix_segments(_segments_from_result(result))

    def transcribe_audio_with_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
    ) -> list[tuple[float, float, str]]:
        """Transcribe audio array and return segments with timestamps."""
        audio = normalize_audio(audio)
        logger.info(f"Transcribing audio with segments: {len(audio)} samples at {sample_rate}Hz")
        result = self._run(audio, timestamps=True)
        return self._fix_segments(_segments_from_result(result))

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
    ) -> str:
        """Transcribe audio from a numpy array."""
        audio = normalize_audio(audio)
        logger.info(f"Transcribing audio buffer: {len(audio)} samples at {sample_rate}Hz")
        result = self._run(audio)
        return self._fix((result.text or "").strip())

    def transcribe_audio_with_words(
        self,
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
    ) -> tuple[str, list[tuple[str, float, float, float]]]:
        """Transcribe audio with word-level timestamps for streaming.

        Returns (text, [(word, start, end, probability)]).  Parakeet does not emit
        per-word probabilities; 1.0 is returned in that slot.
        """
        audio = normalize_audio(audio)
        logger.debug(f"Transcribing with words: {len(audio)} samples at {sample_rate}Hz")
        result = self._run(audio, timestamps=True)

        text = self._fix((result.text or "").strip())
        words: list[tuple[str, float, float, float]] = []
        for stamp in _word_stamps(result):
            words.append(
                (self._fix(stamp["word"]), float(stamp["start"]), float(stamp["end"]), 1.0)
            )
        return text, words

    def transcribe_partial(self, audio: np.ndarray) -> str:
        """Fast partial transcription for streaming use cases."""
        audio = normalize_audio(audio)
        result = self._run(audio)
        return self._fix((result.text or "").strip())

    def _fix_segments(
        self, segments: list[tuple[float, float, str]]
    ) -> list[tuple[float, float, str]]:
        if not self._replacements:
            return segments
        return [(start, end, self._fix(text)) for start, end, text in segments]

    def unload(self) -> None:
        """Unload the model to free memory. No-op if not loaded."""
        self._host.unload()
        self._device = None


def _segments_from_result(result: Any) -> list[tuple[float, float, str]]:
    """Extract (start, end, text) tuples from a NeMo transcription result."""
    timestamp = getattr(result, "timestamp", None) or {}
    segments = timestamp.get("segment") or []
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.get("segment") or seg.get("text") or "").strip()
        if not text:
            continue
        out.append((float(seg["start"]), float(seg["end"]), text))

    if not out and getattr(result, "text", None):
        # Model returned text without segment timestamps — emit a single span.
        out.append((0.0, 0.0, result.text.strip()))
    return out


def _word_stamps(result: Any) -> list[dict]:
    """Extract word-level timestamps from a NeMo transcription result."""
    timestamp = getattr(result, "timestamp", None) or {}
    return timestamp.get("word") or []
