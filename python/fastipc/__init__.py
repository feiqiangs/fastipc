"""fastipc - zero-copy shm + lock-free ring IPC (Linux only)."""
from ._fastipc import Server, Client   # noqa: F401

__all__ = ["Server", "Client"]
