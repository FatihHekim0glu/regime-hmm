# regime-hmm

A from-scratch **Gaussian Hidden Markov Model** for **market-regime
characterization** — fit to index returns to label persistent regimes (low-vol
bull / high-vol bear / crisis), then an **honest** out-of-sample test of whether a
regime-timing exposure overlay beats buy-and-hold after costs.

> **Honest headline.** The HMM cleanly characterizes persistent high/low-vol
> regimes — *that* is the deliverable. The regime-timing exposure overlay does
> **not** reliably beat buy-and-hold out-of-sample after costs, and its in-sample
> edge **decays** once the Deflated Sharpe (with the correct effective `n_trials`)
> is applied. Regime characterization is the win; timing them is not free money.

## The one rule that matters: no smoothed-posterior leakage

The only **tradable** regime signal is the **online forward filter** posterior,
`p(state_t | x_1..x_t)` — it uses data **up to `t` only**. The smoothed
(forward-backward) and Viterbi posteriors condition on the *whole* sample, so they
**peek ahead**: they are in-sample EDA **only** and are never turned into an
out-of-sample label or signal. This is guarded and property-tested
(future-perturbation invariance / prefix-determinism).

## What's in the box

- **`hmm/`** — a pure numpy/scipy Gaussian HMM: emission `kernel` (diag/full
  covariance with a floor), log-space `forward_backward`, Baum-Welch `em` (seeded
  restarts, monotonic log-likelihood), `viterbi` (EDA-only), and the online
  `filter` (the only tradable posterior) with the frozen `HMMModel`.
- **`regimes/`** — `canonicalize` (stable cross-fold state labels) and
  `characterize` (per-regime mean / vol / persistence / duration / drawdown).
- **`backtest/`** — the reused no-lookahead walk-forward engine plus the regime
  `overlay` (filtered-regime exposure, `shift(1)` chokepoint, per-side bps costs,
  cost sensitivity grid) vs buy-and-hold.
- **`evaluation/`** — Probabilistic + Deflated Sharpe (`dsr`),
  Jobson-Korkie-Memmel + block bootstrap (`comparison`), and the pure-function
  timing `verdict` (`no_timing_edge` / `marginal` / `timing_edge`).
- **`data.py`** — a seeded synthetic regime-switch generator (the entire test suite
  runs on it, no network) plus a Polygon-EOD → synthetic loader.
- **`plots.py`** — lazy Plotly figures (regime-shaded series, OOS equity overlay).
- **`cli.py`** — a Typer `fit` / `decode` / `backtest` CLI.

## Install

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
```

`hmmlearn` (in the `dev` extra) is a **parity oracle only** — the test suite checks
the hand-rolled kernel against it to `1e-6`. It is **not** a runtime dependency and
is never imported by `src/` or the deployed API.

## Quality gates

```bash
uv run ruff check src
uv run mypy src
uv run pytest -q --cov=regimehmm --cov-report=term --cov-fail-under=85
```

## Status

Early scaffold: the reused infrastructure is in place and the HMM kernel, regime,
overlay, and verdict modules are typed stubs with full contracts. See
`CHANGELOG.md`. Behaviour, the validation table, limitations (survivorship N/A;
smoothed posteriors non-tradable), and references (Hamilton 1989; Ang-Bekaert 2002;
Rabiner 1989; Bailey-Lopez de Prado DSR) land as the stubs are implemented.

## License

MIT — see `LICENSE`.
