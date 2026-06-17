"""Smoke tests: the package imports cleanly and the public API is exposed.

These guard the scaffold contract for the parallel authors: ``import regimehmm``
must succeed with no import-time side effects, and every name promised in
``__all__`` must be importable (even while its body still raises
``NotImplementedError``). Behavioural tests live in the other partitions and will
be filled in as the stubs are implemented.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    """``import regimehmm`` succeeds and exposes a version string."""
    pkg = importlib.import_module("regimehmm")
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__


@pytest.mark.unit
def test_public_api_is_importable() -> None:
    """Every name in ``regimehmm.__all__`` resolves on the package."""
    pkg = importlib.import_module("regimehmm")
    missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
    assert not missing, f"missing public names: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "regimehmm.hmm.kernel",
        "regimehmm.hmm.forward_backward",
        "regimehmm.hmm.em",
        "regimehmm.hmm.viterbi",
        "regimehmm.hmm.filter",
        "regimehmm.regimes.canonicalize",
        "regimehmm.regimes.characterize",
        "regimehmm.backtest.overlay",
        "regimehmm.evaluation.verdict",
        "regimehmm.data",
        "regimehmm.plots",
        "regimehmm.cli",
    ],
)
def test_submodules_import(module: str) -> None:
    """Each new submodule imports without side effects."""
    assert importlib.import_module(module) is not None
