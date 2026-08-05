"""
Tiny print-based logger for the train detection pipeline.

A single switch — ``configure(debug=True)`` — controls:

* Whether ``.debug(...)`` messages are shown.
* Visual debugging features (cv2.imshow windows, annotated overlays) that
  individual modules gate behind :func:`is_debug`.

Every message is just a ``print()`` to stdout, flushed immediately, so it
always shows up in the terminal interleaved with any third-party output.

Usage::

    from src.debug import configure, get_logger, is_debug

    configure(debug=args.debug)
    log = get_logger(__name__)

    log.info("pipeline started")      # always shown
    log.warning("model missing")      # always shown
    log.debug("per-frame trace")      # shown only when configure(debug=True)

    if is_debug():
        cv2.imshow("Debug", frame)    # heavy visual debug

``--debug`` never hides a message that would normally appear; it only adds
the extra ``.debug(...)`` output on top of the normal info/warning stream.
"""
from __future__ import annotations

import sys

_DEBUG_ENABLED: bool = False


def configure(debug: bool = False) -> None:
    """Enable or disable debug output for the rest of the process.

    Args:
        debug: If ``True``, ``.debug(...)`` calls are printed and visual
            debug features are unlocked. If ``False``, only info / warning /
            error are printed.
    """
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(debug)


def is_debug() -> bool:
    """Return ``True`` iff debug mode is active.

    Gate any heavy/visual debug code path (video playback, image windows,
    per-detection image dumps) behind this check so production runs stay quiet.
    """
    return _DEBUG_ENABLED


class _Logger:
    """Minimal logger: ``.info`` / ``.warning`` / ``.error`` always print,
    ``.debug`` only prints when :func:`configure` was called with ``debug=True``.

    Supports printf-style formatting (``log.info("x=%d", 5)``) so the existing
    call sites work unchanged.
    """

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _emit(self, level: str, msg: str, args: tuple) -> None:
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = f"{msg} {args}"
        print(f"[{level}] [{self.name}] {msg}", flush=True)

    def info(self, msg: str, *args) -> None:
        """Print an informational message — always visible."""
        self._emit("INFO", msg, args)

    def warning(self, msg: str, *args) -> None:
        """Print a warning — always visible."""
        self._emit("WARN", msg, args)

    def error(self, msg: str, *args) -> None:
        """Print an error message — always visible."""
        self._emit("ERROR", msg, args)

    def debug(self, msg: str, *args) -> None:
        """Print a debug trace — visible only under ``configure(debug=True)``."""
        if _DEBUG_ENABLED:
            self._emit("DEBUG", msg, args)


def get_logger(name: str) -> _Logger:
    """Return a module-scoped logger.

    Prefer ``get_logger(__name__)`` so messages are prefixed with the module
    path — that makes the output readable when multiple stages interleave.
    """
    return _Logger(name)
