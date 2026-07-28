"""SynTrustBench executable tabular protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .models import BenchmarkResult

__all__ = ["BenchmarkResult", "__version__", "evaluate"]


def evaluate(*args: Any, **kwargs: Any):
    """Lazily import and run the executable protocol."""

    from .engine import evaluate as run

    return run(*args, **kwargs)


def __getattr__(name: str):
    if name == "BenchmarkResult":
        from .models import BenchmarkResult

        return BenchmarkResult
    raise AttributeError(name)
