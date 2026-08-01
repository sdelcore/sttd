"""Tests for the OpenAI-compatible request/response helpers in http_server."""

import shutil

import numpy as np
import pytest

from voiced.audio_codec import audio_to_wav
from voiced.http_server import (
    AudioDecodeError,
    decode_audio,
    parse_multipart_fields,
    suffix_for_content_type,
    transcode_wav,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def multipart_body(boundary: str, parts: list[tuple[str, bytes, str | None]]) -> bytes:
    """Build a multipart/form-data body from ``(name, value, filename)`` triples."""
    chunks = []
    for name, value, filename in parts:
        disposition = f'form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: {disposition}\r\n\r\n".encode() + value + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


class TestSuffixForContentType:
    def test_known_type(self):
        assert suffix_for_content_type("audio/webm") == ".webm"

    def test_parameters_are_ignored(self):
        assert suffix_for_content_type("audio/webm; codecs=opus") == ".webm"

    def test_case_is_ignored(self):
        assert suffix_for_content_type("Audio/FLAC") == ".flac"

    def test_unknown_type_falls_back_to_wav(self):
        assert suffix_for_content_type("application/octet-stream") == ".wav"


class TestParseMultipartFields:
    def test_file_part_keeps_bytes_and_filename(self):
        body = multipart_body("abc123", [("file", b"\x00\x01audio", "speech.webm")])
        fields = parse_multipart_fields(body, "multipart/form-data; boundary=abc123")
        assert fields["file"] == (b"\x00\x01audio", "speech.webm")

    def test_text_and_file_parts_together(self):
        body = multipart_body(
            "xyz",
            [
                ("model", b"whisper-1", None),
                ("response_format", b"text", None),
                ("file", b"RIFF", "a.wav"),
            ],
        )
        fields = parse_multipart_fields(body, "multipart/form-data; boundary=xyz")
        assert fields["model"][0] == b"whisper-1"
        assert fields["response_format"][0] == b"text"
        assert fields["file"][1] == "a.wav"

    def test_non_multipart_content_type_raises(self):
        with pytest.raises(ValueError):
            parse_multipart_fields(b"{}", "application/json")


class TestTranscodeWav:
    def test_wav_is_returned_unchanged(self):
        wav = audio_to_wav(np.zeros(1000, dtype=np.float32), 24000)
        assert transcode_wav(wav, "wav") is wav

    @needs_ffmpeg
    def test_mp3_is_re_encoded(self):
        wav = audio_to_wav(np.sin(np.linspace(0, 200, 24000)).astype(np.float32), 24000)
        mp3 = transcode_wav(wav, "mp3")
        assert len(mp3) > 0
        assert not mp3.startswith(b"RIFF")

    @needs_ffmpeg
    def test_pcm_has_no_container_header(self):
        wav = audio_to_wav(np.zeros(2400, dtype=np.float32), 24000)
        pcm = transcode_wav(wav, "pcm")
        assert not pcm.startswith(b"RIFF")


class TestDecodeAudio:
    def test_wav_round_trip(self):
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        decoded, sample_rate = decode_audio(audio_to_wav(audio, 16000), ".wav")
        assert sample_rate == 16000
        assert decoded.shape == audio.shape
        np.testing.assert_allclose(decoded, audio, atol=1e-4)

    def test_stereo_is_mixed_to_mono(self):
        import io

        import soundfile as sf

        stereo = np.zeros((800, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        stereo[:, 1] = -0.5
        buffer = io.BytesIO()
        sf.write(buffer, stereo, 16000, format="WAV", subtype="PCM_16")

        decoded, _ = decode_audio(buffer.getvalue(), ".wav")
        assert decoded.ndim == 1
        np.testing.assert_allclose(decoded, np.zeros(800), atol=1e-4)

    @needs_ffmpeg
    def test_encoded_input_is_resampled_to_16k(self):
        wav = audio_to_wav(np.zeros(48000, dtype=np.float32), 48000)
        flac = transcode_wav(wav, "flac")
        decoded, sample_rate = decode_audio(flac, ".flac")
        assert sample_rate == 16000
        assert decoded.ndim == 1

    def test_garbage_raises_audio_decode_error(self):
        with pytest.raises(AudioDecodeError):
            decode_audio(b"not audio at all", ".wav")
