"""HTTP client for transcription requests."""

import io
import json
import logging
import wave
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)


class ServerError(Exception):
    """Server returned an error."""

    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.code = code


class ConnectionError(Exception):
    """Could not connect to server."""

    pass


class TimeoutError(Exception):
    """Request timed out."""

    pass


def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes.

    Args:
        audio: Audio data as numpy array (float32, mono).
        sample_rate: Sample rate of the audio.

    Returns:
        WAV file bytes.
    """
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


class TranscriptionClient:
    """HTTP client for remote transcription."""

    def __init__(self, server_url: str, timeout: float = 60.0):
        """Initialize the transcription client.

        Args:
            server_url: Base URL of the transcription server.
            timeout: Request timeout in seconds.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> str:
        """Send audio to server for transcription.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio.
            language: Optional language code to use.

        Returns:
            Transcribed text.

        Raises:
            ServerError: If server returned an error.
            ConnectionError: If could not connect to server.
            TimeoutError: If request timed out.
        """
        wav_bytes = audio_to_wav(audio, sample_rate)

        url = f"{self.server_url}/transcribe"
        if language:
            url += f"?language={language}"

        logger.info(f"Sending {len(wav_bytes)} bytes to {url}")

        req = Request(
            url,
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )

        try:
            response = urlopen(req, timeout=self.timeout)
            result = json.loads(response.read().decode("utf-8"))
            logger.info(
                f"Transcription completed: {result.get('processing_time', 0):.1f}s processing time"
            )
            return result["text"]

        except HTTPError as e:
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                raise ServerError(
                    error_body.get("error", str(e)), error_body.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise ServerError(str(e), "HTTP_ERROR")

        except URLError as e:
            if "timed out" in str(e.reason).lower():
                raise TimeoutError(f"Request timed out after {self.timeout}s")
            raise ConnectionError(f"Could not connect to server: {e.reason}")

        except TimeoutError:
            raise TimeoutError(f"Request timed out after {self.timeout}s")

    def health_check(self) -> dict:
        """Check server health.

        Returns:
            Health status dictionary with model, device, etc.

        Raises:
            ConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/health"

        try:
            response = urlopen(url, timeout=5.0)
            return json.loads(response.read().decode("utf-8"))
        except URLError as e:
            raise ConnectionError(f"Could not connect to server: {e.reason}")
        except Exception as e:
            raise ConnectionError(f"Health check failed: {e}")

    def get_status(self) -> dict:
        """Get detailed server status.

        Returns:
            Status dictionary with model, device, uptime, request count, etc.

        Raises:
            ConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/status"

        try:
            response = urlopen(url, timeout=5.0)
            return json.loads(response.read().decode("utf-8"))
        except URLError as e:
            raise ConnectionError(f"Could not connect to server: {e.reason}")
        except Exception as e:
            raise ConnectionError(f"Status check failed: {e}")

    def is_available(self) -> bool:
        """Check if server is available.

        Returns:
            True if server is reachable and healthy.
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False
