"""Integration tests for the Typer CLI (``regimehmm.cli``).

Two layers:

* The orchestration functions (:func:`~regimehmm.cli.fit`,
  :func:`~regimehmm.cli.decode`) are exercised DIRECTLY on the deterministic
  synthetic regime-switch generator - a tiny offline fit -> decode smoke run that
  proves the pipeline wires up end-to-end and exits cleanly. These need no Typer.
* The Typer app surface (``build_app``, ``--help``, a ``decode`` CliRunner
  invocation) is exercised through :class:`typer.testing.CliRunner` when Typer is
  installed, and SKIPPED otherwise (Typer is imported lazily, so importing the
  module and running the core pipeline never requires it).

Everything runs OFFLINE on the synthetic generator; nothing touches the network.
"""

from __future__ import annotations

import pytest

from regimehmm import cli

pytestmark = pytest.mark.integration

# A tiny synthetic window keeps the fit fast while still exercising every stage.
_SMOKE_KWARGS = {"n_states": 2, "feature_set": "returns", "n_obs": 300, "seed": 7}


# --------------------------------------------------------------------------- #
# Core orchestration (no Typer required)                                       #
# --------------------------------------------------------------------------- #
def test_fit_smoke_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``fit`` runs the synthetic fit pipeline offline and returns exit code 0."""
    code = cli.fit(**_SMOKE_KWARGS)
    assert code == 0
    out = capsys.readouterr().out
    assert "regime-hmm fit" in out
    assert "transition matrix" in out
    assert "data source        : synthetic" in out


def test_decode_smoke_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``decode`` fits then decodes via the ONLINE FILTER and returns exit code 0."""
    code = cli.decode(**_SMOKE_KWARGS)
    assert code == 0
    out = capsys.readouterr().out
    assert "ONLINE FILTER" in out
    # The per-regime characterization table prints one row per state (0 and 1).
    assert "    0  " in out
    assert "    1  " in out


def test_fit_then_decode_are_consistent(capsys: pytest.CaptureFixture[str]) -> None:
    """A fit -> decode sequence on the same seed both succeed (the brief's smoke run)."""
    assert cli.fit(**_SMOKE_KWARGS) == 0
    capsys.readouterr()
    assert cli.decode(**_SMOKE_KWARGS) == 0


def test_fit_rejects_bad_feature_set(capsys: pytest.CaptureFixture[str]) -> None:
    """An unsupported feature set is handled as a library error (exit code 1)."""
    code = cli.fit(n_states=2, feature_set="nope", n_obs=200, seed=7)
    assert code == 1
    assert "error:" in capsys.readouterr().out


def test_fit_three_state_feature_vol() -> None:
    """A 3-state fit on the volatility feature set also completes cleanly."""
    assert cli.fit(n_states=3, feature_set="returns_vol", n_obs=400, seed=7) == 0


def test_decode_rejects_bad_feature_set(capsys: pytest.CaptureFixture[str]) -> None:
    """``decode`` surfaces a library error (bad feature set) as exit code 1."""
    code = cli.decode(n_states=2, feature_set="nope", n_obs=200, seed=7)
    assert code == 1
    assert "error:" in capsys.readouterr().out


def test_backtest_smoke_run_is_honest_null(capsys: pytest.CaptureFixture[str]) -> None:
    """``backtest`` runs the overlay-vs-buy-and-hold pipeline and prints a verdict.

    On the synthetic generator the honest outcome is that the regime-timing overlay
    does NOT beat buy-and-hold after costs, so the verdict must be ``no_timing_edge``
    and the effective trial count must be the full grid product (3 x 3 x 4 = 36),
    never silently collapsed to 1.
    """
    code = cli.backtest(**_SMOKE_KWARGS, cost_bps=10.0)
    assert code == 0
    out = capsys.readouterr().out
    assert "verdict            : no_timing_edge" in out
    assert "effective n_trials : 36" in out
    assert "Memmel-JK p-value" in out
    assert "deflated Sharpe" in out


def test_backtest_rejects_bad_feature_set(capsys: pytest.CaptureFixture[str]) -> None:
    """``backtest`` surfaces a library error (bad feature set) as exit code 1."""
    code = cli.backtest(n_states=2, feature_set="nope", n_obs=200, cost_bps=10.0, seed=7)
    assert code == 1
    assert "error:" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Typer app surface (skipped when Typer is not installed)                      #
# --------------------------------------------------------------------------- #
def test_build_app_help_lists_commands() -> None:
    """``--help`` lists all three sub-commands (fit, decode, backtest)."""
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    app = cli.build_app()
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fit" in result.stdout
    assert "decode" in result.stdout
    assert "backtest" in result.stdout


def test_cli_decode_smoke_via_runner() -> None:
    """Invoking ``decode`` through the Typer runner exits 0 on the synthetic data."""
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    app = cli.build_app()
    result = CliRunner().invoke(
        app,
        ["decode", "--n-states", "2", "--feature-set", "returns", "--n-obs", "300"],
    )
    assert result.exit_code == 0
    assert "ONLINE FILTER" in result.stdout


def test_cli_fit_smoke_via_runner() -> None:
    """Invoking ``fit`` through the Typer runner exits 0 and prints the params."""
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    app = cli.build_app()
    result = CliRunner().invoke(
        app,
        ["fit", "--n-states", "2", "--feature-set", "returns", "--n-obs", "300"],
    )
    assert result.exit_code == 0
    assert "regime-hmm fit" in result.stdout


def test_cli_backtest_smoke_via_runner() -> None:
    """Invoking ``backtest`` through the Typer runner exits 0 and prints the verdict."""
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    app = cli.build_app()
    result = CliRunner().invoke(
        app,
        ["backtest", "--n-states", "2", "--feature-set", "returns", "--n-obs", "300"],
    )
    assert result.exit_code == 0
    assert "verdict" in result.stdout
