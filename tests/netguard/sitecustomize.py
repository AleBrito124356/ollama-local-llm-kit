"""Block every non-loopback network connection in this Python process.

The test-suite imports this module directly, and subprocesses started by the
tests pick it up automatically because ``tests/netguard`` is put first on their
``PYTHONPATH`` (Python imports ``sitecustomize`` at startup). Anything that tries
to reach NVIDIA, a real Ollama on another host, or any other remote service fails
loudly instead of silently making a paid or remote call.
"""

from __future__ import annotations

import socket

LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1"}


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").lower()
    return host in LOOPBACK_NAMES or host.startswith("127.")


class NetworkBlocked(OSError):
    """Raised when code under test tries to leave the machine."""


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _check(address) -> None:
    if isinstance(address, tuple) and address and not _is_loopback(address[0]):
        raise NetworkBlocked(f"non-loopback network access is blocked in tests: {address!r}")


def _guarded_connect(self, address):
    _check(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(address)
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host, *args, **kwargs):
    if host is not None and not _is_loopback(host):
        raise NetworkBlocked(f"DNS lookups of non-loopback hosts are blocked in tests: {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


def install() -> None:
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo


install()
