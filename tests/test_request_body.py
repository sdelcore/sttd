"""Tests for request body framing — Content-Length and Transfer-Encoding: chunked."""

import io

import pytest

from voiced.http_server import (
    BodyTooLargeError,
    MalformedBodyError,
    TranscriptionHandler,
    read_chunked_body,
)


def chunked(*chunks: bytes, trailers: bytes = b"") -> bytes:
    """Encode ``chunks`` as an HTTP chunked body."""
    out = b""
    for chunk in chunks:
        out += f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n"
    return out + b"0\r\n" + trailers + b"\r\n"


class FakeHandler:
    """A TranscriptionHandler with the socket plumbing replaced by buffers."""

    def __init__(self, body: bytes, headers: dict[str, str]):
        self.handler = TranscriptionHandler.__new__(TranscriptionHandler)
        self.handler.rfile = io.BytesIO(body)
        self.handler.headers = headers
        self.handler.close_connection = False
        self.handler._body_consumed = False
        self.errors: list[tuple[int, str, str]] = []
        self.handler._send_error_json = self._record

    def _record(self, status: int, message: str, code: str) -> None:
        if self.handler._request_has_body() and not self.handler._body_consumed:
            self.handler.close_connection = True
        self.errors.append((status, message, code))

    def read(self, max_bytes: int = 1024 * 1024):
        return self.handler._read_body(max_bytes)


class TestReadChunkedBody:
    def test_single_chunk(self):
        assert read_chunked_body(io.BytesIO(chunked(b"hello")), 1024) == b"hello"

    def test_chunks_are_joined_in_order(self):
        body = chunked(b"abc", b"def", b"ghi")
        assert read_chunked_body(io.BytesIO(body), 1024) == b"abcdefghi"

    def test_binary_payload_survives_intact(self):
        payload = bytes(range(256)) * 8
        assert read_chunked_body(io.BytesIO(chunked(payload)), 1 << 20) == payload

    def test_empty_body(self):
        assert read_chunked_body(io.BytesIO(chunked()), 1024) == b""

    def test_chunk_extensions_are_ignored(self):
        body = b"5;name=value\r\nhello\r\n0\r\n\r\n"
        assert read_chunked_body(io.BytesIO(body), 1024) == b"hello"

    def test_trailers_are_drained(self):
        body = chunked(b"hi", trailers=b"X-Checksum: abc\r\n")
        stream = io.BytesIO(body)
        assert read_chunked_body(stream, 1024) == b"hi"
        assert stream.read() == b""

    def test_stream_is_left_at_the_end_of_the_body(self):
        stream = io.BytesIO(chunked(b"hi") + b"LEFTOVER")
        read_chunked_body(stream, 1024)
        assert stream.read() == b"LEFTOVER"

    def test_size_over_limit_raises(self):
        with pytest.raises(BodyTooLargeError):
            read_chunked_body(io.BytesIO(chunked(b"x" * 100)), 10)

    def test_limit_counts_all_chunks_together(self):
        with pytest.raises(BodyTooLargeError):
            read_chunked_body(io.BytesIO(chunked(b"x" * 6, b"x" * 6)), 10)

    def test_truncated_body_raises(self):
        with pytest.raises(MalformedBodyError):
            read_chunked_body(io.BytesIO(b"5\r\nhel"), 1024)

    def test_missing_terminating_chunk_raises(self):
        with pytest.raises(MalformedBodyError):
            read_chunked_body(io.BytesIO(b"5\r\nhello\r\n"), 1024)

    def test_non_hex_chunk_size_raises(self):
        with pytest.raises(MalformedBodyError):
            read_chunked_body(io.BytesIO(b"zz\r\nhello\r\n0\r\n\r\n"), 1024)

    def test_chunk_without_crlf_terminator_raises(self):
        with pytest.raises(MalformedBodyError):
            read_chunked_body(io.BytesIO(b"5\r\nhelloXX0\r\n\r\n"), 1024)


class TestReadBody:
    def test_content_length_body(self):
        h = FakeHandler(b"hello", {"Content-Length": "5"})
        assert h.read() == b"hello"
        assert h.errors == []

    def test_chunked_body(self):
        h = FakeHandler(chunked(b"hello"), {"Transfer-Encoding": "chunked"})
        assert h.read() == b"hello"
        assert h.errors == []

    def test_chunked_is_detected_case_insensitively(self):
        h = FakeHandler(chunked(b"hi"), {"Transfer-Encoding": "Chunked"})
        assert h.read() == b"hi"

    def test_chunked_wins_over_a_stale_content_length(self):
        h = FakeHandler(chunked(b"hello"), {"Transfer-Encoding": "chunked", "Content-Length": "0"})
        assert h.read() == b"hello"

    def test_no_body_reads_as_empty(self):
        h = FakeHandler(b"", {})
        assert h.read() == b""
        assert h.errors == []

    def test_oversize_content_length_is_rejected(self):
        h = FakeHandler(b"x" * 100, {"Content-Length": "100"})
        assert h.read(10) is None
        assert h.errors[0][0] == 413

    def test_oversize_chunked_body_is_rejected(self):
        h = FakeHandler(chunked(b"x" * 100), {"Transfer-Encoding": "chunked"})
        assert h.read(10) is None
        assert h.errors[0][0] == 413

    def test_malformed_chunked_body_is_rejected(self):
        h = FakeHandler(b"zz\r\n", {"Transfer-Encoding": "chunked"})
        assert h.read() is None
        assert h.errors[0][0] == 400

    def test_invalid_content_length_is_rejected(self):
        h = FakeHandler(b"hello", {"Content-Length": "abc"})
        assert h.read() is None
        assert h.errors[0][0] == 400

    def test_short_body_is_rejected(self):
        h = FakeHandler(b"hi", {"Content-Length": "10"})
        assert h.read() is None
        assert h.errors[0][0] == 400


class TestConnectionClose:
    """An error sent before the body is read must not leave bytes in the socket.

    Otherwise keep-alive parses the remainder as the next request, which is how
    an unread chunked upload surfaced as "Bad request syntax ('24')".
    """

    def test_rejecting_an_unread_chunked_body_closes_the_connection(self):
        h = FakeHandler(chunked(b"x" * 100), {"Transfer-Encoding": "chunked"})
        h.read(10)
        assert h.handler.close_connection is True

    def test_rejecting_an_unread_sized_body_closes_the_connection(self):
        h = FakeHandler(b"x" * 100, {"Content-Length": "100"})
        h.read(10)
        assert h.handler.close_connection is True

    def test_a_fully_read_body_keeps_the_connection_open(self):
        h = FakeHandler(chunked(b"hello"), {"Transfer-Encoding": "chunked"})
        h.read()
        h.handler._send_error_json(400, "bad audio", "INVALID_AUDIO")
        assert h.handler.close_connection is False

    def test_a_bodyless_request_keeps_the_connection_open(self):
        h = FakeHandler(b"", {})
        h.handler._send_error_json(404, "Not found", "NOT_FOUND")
        assert h.handler.close_connection is False
