"""HTTP server for transcription requests."""

import io
import json
import logging
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from sttd.config import Config, load_config
from sttd.transcriber import Transcriber

logger = logging.getLogger(__name__)


def wav_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert WAV bytes to numpy array and sample rate."""
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

        if sample_width == 2:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
            audio = audio_float32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = np.mean(audio, axis=1)

    return audio, sample_rate


class TranscriptionHandler(BaseHTTPRequestHandler):
    """Handle transcription HTTP requests."""

    transcriber: Transcriber
    config: Config
    start_time: float
    request_count: int = 0
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, code: str) -> None:
        self._send_json(status, {"error": message, "code": code})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._handle_health()
        elif path == "/status":
            self._handle_status()
        else:
            self._send_error_json(404, "Not found", "NOT_FOUND")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/transcribe":
            self._handle_transcribe(parsed.query)
        else:
            self._send_error_json(404, "Not found", "NOT_FOUND")

    def _handle_health(self) -> None:
        device = self.transcriber._get_device()
        self._send_json(
            200,
            {
                "status": "healthy",
                "model": self.transcriber.config.model,
                "device": device,
            },
        )

    def _handle_status(self) -> None:
        device = self.transcriber._get_device()
        uptime = time.time() - self.start_time
        self._send_json(
            200,
            {
                "status": "ok",
                "state": "idle",
                "model": self.transcriber.config.model,
                "device": device,
                "language": self.transcriber.config.language,
                "request_count": TranscriptionHandler.request_count,
                "uptime_seconds": round(uptime, 1),
            },
        )

    def _handle_transcribe(self, query_string: str) -> None:
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length == 0:
            self._send_error_json(400, "No audio data provided", "NO_AUDIO")
            return

        if content_length > 100 * 1024 * 1024:
            self._send_error_json(413, "Audio file too large (max 100MB)", "AUDIO_TOO_LARGE")
            return

        query_params = parse_qs(query_string)
        language = query_params.get("language", [None])[0]

        wav_bytes = self.rfile.read(content_length)

        try:
            audio, sample_rate = wav_to_audio(wav_bytes)
        except Exception as e:
            logger.error(f"Failed to parse WAV: {e}")
            self._send_error_json(400, f"Invalid WAV format: {e}", "INVALID_AUDIO")
            return

        duration = len(audio) / sample_rate
        logger.info(f"Transcribing {duration:.1f}s of audio at {sample_rate}Hz")

        try:
            original_language = self.transcriber.config.language
            if language:
                self.transcriber.config.language = language

            start_time = time.time()
            text = self.transcriber.transcribe_audio(audio, sample_rate)
            elapsed = time.time() - start_time

            if language:
                self.transcriber.config.language = original_language

            TranscriptionHandler.request_count += 1

            self._send_json(
                200,
                {
                    "text": text,
                    "duration": round(duration, 2),
                    "language": language or original_language,
                    "processing_time": round(elapsed, 2),
                },
            )
        except Exception as e:
            logger.exception(f"Transcription failed: {e}")
            self._send_error_json(500, f"Transcription failed: {e}", "TRANSCRIPTION_ERROR")


class TranscriptionServer:
    """HTTP server wrapper for transcription service."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        config: Config | None = None,
    ):
        self.config = config or load_config()
        self.host = host or self.config.server.host
        self.port = port or self.config.server.port
        self.transcriber = Transcriber(self.config.transcription)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def _preload_model(self) -> None:
        logger.info("Pre-loading transcription model...")
        _ = self.transcriber.model
        logger.info("Model loaded successfully")

    def start(self, preload: bool = True) -> None:
        """Start the HTTP server."""
        if self._running:
            return

        if preload:
            self._preload_model()

        TranscriptionHandler.transcriber = self.transcriber
        TranscriptionHandler.config = self.config
        TranscriptionHandler.start_time = time.time()
        TranscriptionHandler.request_count = 0

        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            TranscriptionHandler,
        )
        self._running = True

        logger.info(f"Starting HTTP server on {self.host}:{self.port}")
        self._server.serve_forever()

    def start_background(self, preload: bool = True) -> None:
        """Start the HTTP server in a background thread."""
        if self._running:
            return

        if preload:
            self._preload_model()

        TranscriptionHandler.transcriber = self.transcriber
        TranscriptionHandler.config = self.config
        TranscriptionHandler.start_time = time.time()
        TranscriptionHandler.request_count = 0

        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            TranscriptionHandler,
        )
        self._running = True

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"HTTP server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the HTTP server."""
        if not self._running:
            return

        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        self.transcriber.unload()
        logger.info("HTTP server stopped")
