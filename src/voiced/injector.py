"""Text injection using clipboard."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def is_clipboard_available() -> bool:
    """Check if wl-copy is available."""
    return shutil.which("wl-copy") is not None


def _get_wayland_display() -> str | None:
    """Get WAYLAND_DISPLAY, auto-detecting if unset.

    Returns:
        The Wayland display name (e.g., 'wayland-1') or None if not found.
    """
    display = os.environ.get("WAYLAND_DISPLAY")
    if display:
        return display

    # Auto-detect: look for wayland-* sockets in XDG_RUNTIME_DIR
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    runtime_path = Path(runtime_dir)

    if not runtime_path.exists():
        return None

    # Find first wayland socket (sorted to prefer wayland-0, wayland-1, etc.)
    for entry in sorted(runtime_path.glob("wayland-[0-9]")):
        if entry.is_socket():
            return entry.name

    return None


def inject_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using wl-copy.

    Args:
        text: Text to copy to clipboard.

    Returns:
        True if successful, False otherwise.
    """
    if not text:
        return True

    if not is_clipboard_available():
        logger.error("wl-copy is not available")
        return False

    try:
        # Build environment with auto-detected WAYLAND_DISPLAY
        env = os.environ.copy()
        wayland_display = _get_wayland_display()
        if wayland_display:
            env["WAYLAND_DISPLAY"] = wayland_display
            logger.debug(f"Using Wayland display: {wayland_display}")

        result = subprocess.run(
            ["wl-copy", "--", text],
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        )
        if result.returncode != 0:
            logger.error(f"wl-copy failed: {result.stderr}")
            return False
        logger.info("Text copied to clipboard")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Clipboard operation timed out")
        return False
    except Exception as e:
        logger.error(f"Clipboard error: {e}")
        return False
