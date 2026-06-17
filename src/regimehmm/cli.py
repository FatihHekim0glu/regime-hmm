"""Command-line interface (Typer).

A thin orchestration layer over the compute library: generate or load a return
series, fit the HMM, decode regimes, characterize them, and run the honest
overlay-vs-buy-and-hold backtest. Typer is imported LAZILY inside :func:`build_app`
so importing this module registers no commands and performs no I/O (no import-time
side effects). The ``regime-hmm`` console-script entry point calls :func:`app`.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``fit``, ``decode``, ``backtest``) on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so that
    importing :mod:`regimehmm.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.

    Raises
    ------
    NotImplementedError
        Until the CLI commands are implemented.
    """
    raise NotImplementedError


def app() -> None:
    """Console-script entry point for the ``regime-hmm`` command.

    Builds the Typer app via :func:`build_app` and invokes it. Referenced by
    ``[project.scripts]`` in ``pyproject.toml``.

    Raises
    ------
    NotImplementedError
        Until :func:`build_app` is implemented.
    """
    build_app()()
