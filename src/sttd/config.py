"""Configuration management for sttd."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class TranscriptionConfig:
    """Transcription settings."""

    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "en"
    streaming: bool = True
    chunk_duration: float = 2.0
    max_window: float = 30.0  # Max seconds of audio in sliding window


@dataclass
class AudioConfig:
    """Audio capture settings."""

    sample_rate: int = 16000
    channels: int = 1
    device: str = "default"
    beep_enabled: bool = True


@dataclass
class OutputConfig:
    """Output settings."""

    method: str = "wtype"  # wtype, clipboard, both


@dataclass
class Config:
    """Main configuration container."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg_config) / "sttd" / "config.toml"


def get_cache_dir() -> Path:
    """Get the cache directory for sttd."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(xdg_cache) / "sttd"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_socket_path() -> Path:
    """Get the path to the Unix domain socket."""
    return get_cache_dir() / "control.sock"


def get_pid_path() -> Path:
    """Get the path to the PID file."""
    return get_cache_dir() / "daemon.pid"


def load_config() -> Config:
    """Load configuration from file, falling back to defaults."""
    config_path = get_config_path()

    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    config = Config()

    if "transcription" in data:
        for key, value in data["transcription"].items():
            if hasattr(config.transcription, key):
                setattr(config.transcription, key, value)

    if "audio" in data:
        for key, value in data["audio"].items():
            if hasattr(config.audio, key):
                setattr(config.audio, key, value)

    if "output" in data:
        for key, value in data["output"].items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)

    return config


def save_default_config() -> None:
    """Save a default configuration file."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = """\
[transcription]
model = "base"           # tiny, base, small, medium, large-v3
device = "auto"          # auto, cuda, cpu
compute_type = "auto"    # auto, float16, int8, float32
language = "en"
streaming = true
chunk_duration = 2.0     # Seconds per chunk
max_window = 30.0        # Max seconds in sliding window

[audio]
sample_rate = 16000
channels = 1
device = "default"       # or specific device name
beep_enabled = true      # audio feedback on start/stop

[output]
method = "wtype"         # wtype, clipboard, both
"""
    with open(config_path, "w") as f:
        f.write(default_config)
