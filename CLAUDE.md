# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Enter development environment (NixOS with direnv)
direnv allow

# Or manually with nix
nix develop

# Run the daemon
voiced start              # Foreground
voiced start --daemon     # Background

# STT commands
voiced toggle             # Toggle recording
voiced status             # Check daemon state
voiced stop               # Stop daemon
voiced transcribe file.wav  # Transcribe audio file

# TTS commands
voiced speak "Hello world"           # Speak text
voiced speak --stream "Hello world"  # Low-latency streaming
voiced speak --clipboard             # Speak clipboard contents
voiced voices list                   # List available voices

# Lint
ruff check src/
ruff format src/

# Run tests
pytest tests/ -v
pytest tests/test_cli.py -v          # Single test file
pytest tests/test_cli.py::test_name  # Single test
```

## Architecture

voiced is a voice daemon for Wayland/Hyprland that provides both:
- **STT (Speech-to-Text)**: Using NVIDIA Parakeet-TDT (NeMo) for transcription
- **TTS (Text-to-Speech)**: Using Kokoro-82M for synthesis

### Component Flow

```
CLI Command → Unix Socket IPC → Daemon
                                  ├── Server (server.py) - Unix socket handler
                                  ├── Recorder (recorder.py) - sounddevice audio capture
                                  ├── WorkerHost (worker_host.py) - Inference Worker supervisor + proxies
                                  ├── Injector (injector.py) - wl-clipboard text injection
                                  └── TrayIcon (tray.py) - D-Bus StatusNotifierItem
                                        │ pipe IPC (spawn)
                                  Inference Worker (worker.py) - disposable child process
                                  ├── Transcriber (transcriber.py) - Parakeet-TDT STT
                                  ├── Synthesizer (synthesizer.py) - Kokoro TTS
                                  └── Diarizer (diarizer.py) - SpeechBrain speaker ID
```

### Key Design Patterns

**Record-then-Transcribe (STT)**: Toggle once to start recording (RED tray icon), toggle again to stop and begin batch transcription (YELLOW icon). When complete (BLUE icon), text is copied to clipboard. Starting a recording also warms the Inference Worker in the background so the stop-toggle doesn't pay worker-spawn + model-load latency.

**Disposable Inference Worker**: STT/TTS inference runs in a child process spawned lazily on the first request (`worker_host.py` parent side, `worker.py` child side). After the shared idle timeout (default 15 min) with no active operations, the worker process is terminated — process exit is what reliably releases VRAM; `torch.cuda.empty_cache()` in a live process does not. The next request transparently starts a fresh worker. The parent process must never import torch/NeMo/Kokoro (guarded by `tests/test_parent_imports.py`).

A worker that dies mid-request is retried once on a fresh process (`WorkerHost.request`; streams only retry before the first chunk reaches the consumer), so a crash costs latency instead of the request.

**Serialized GPU inference (`gpu.py`)**: Every call into a loaded model runs inside `gpu_exclusive()`. This is a correctness requirement, not a performance tuning knob — read `src/voiced/gpu.py` before touching any model call site.

Two hazards force it, and neither is a backup for the other:

1. *NeMo is not re-entrant.* `transcribe()` freezes the encoder/decoder/joint and stashes `training`, `dither`, and `pad_to` on the module, then restores them on exit. Two overlapping calls on one model tear down each other's state. This races on CPU too. Upstream has no lock of its own (NVIDIA/NeMo#15771).
2. *CUDA graph capture is process-scoped.* While a capture is underway no other thread in the process may issue CUDA work, and a lock cannot cover work outside model calls (the cyclic GC frees tensors on arbitrary threads). So `transcriber._load_model` **disables** NeMo's graph decoder outright and raises if it cannot. Do not "optimise" that back on: NeMo captures with `capture_error_mode="thread_local"`, which turns CUDA's cross-thread check *off*, so a violation is undefined behaviour rather than an error.

Before this existed, two concurrent requests returned `HTTP 200` with a **silently wrong transcript**, and a third poisoned the CUDA graph so every later request failed until the worker restarted. `tests/test_concurrency_stt.py` guards it against a golden baseline and includes a negative control that strips the lock and asserts the corruption returns.

Lock ordering: `gpu_exclusive()` is innermost. Taking `ModelHost._lock` then the GPU lock is correct; the reverse deadlocks.

**Native library load order**: `worker.preload_native_stack()` imports `pyarrow.dataset` and `lhotse` on the worker's main thread before any model loads. The STT stack (NeMo → lhotse → pyarrow) segfaults inside Arrow's mimalloc allocator when it is loaded into a process that already holds Kokoro and torch; the reverse order is safe. Without the preload, request order decided library order and a TTS-then-STT worker died.

**HTTP Client-Server Mode**: For remote STT/TTS:
- `http_server.py` - HTTP server with `/transcribe`, `/synthesize`, `/health` endpoints
- `http_client.py` - HTTP client for remote connections
- WebSocket support for streaming TTS at `/synthesize/stream`
- OpenAI-compatible aliases at `/v1/audio/speech`, `/v1/audio/transcriptions` and `/v1/audio/voices`. They translate to the same handlers, so a client written against the OpenAI audio API (Open WebUI, the OpenAI SDK) works unchanged. The native routes stay the richer ones — only they return segments and speaker IDs.

**State Machine**: Daemon states are `IDLE → RECORDING → TRANSCRIBING → IDLE`. The tray icon reflects state visually.

**IPC Protocol**: JSON over Unix domain socket at `~/.cache/voiced/control.sock`. Commands: `toggle`, `status`, `stop`.

**GPU Detection**: Uses `torch.cuda.is_available()` for CUDA detection (NeMo is torch-native).

### Config Locations

- Config: `~/.config/voiced/config.toml`
- Socket: `~/.cache/voiced/control.sock`
- PID: `~/.cache/voiced/daemon.pid`
- Voice presets: `~/.cache/voiced/voices/`

## Code Style

- Line length: 100 chars
- Ruff linting with rules: E, F, I, N, W, UP
- Type hints throughout
- Minimal comments - code should be self-explanatory
