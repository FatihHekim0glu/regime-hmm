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
- **`analysis.py`** — `run_regime_analysis(...)`, the single end-to-end entrypoint
  the hosted backend calls (fit → canonicalize → online-filter decode →
  characterize → overlay-vs-buy-and-hold → Memmel-JK + Deflated Sharpe → honest
  verdict), plus `assemble_regime_figures(...)` for the two frontend Plotly figures.
- **`cli.py`** — a Typer `fit` / `decode` / `backtest` CLI.

## Public entrypoint

```python
from regimehmm import run_regime_analysis, assemble_regime_figures

# In-process (a return Series) or load-at-request (ticker → Polygon, synthetic fallback):
result = run_regime_analysis(returns, n_states=3, feature_set="returns_vol",
                             cost_bps=10, seed=7)
summary = result.summary   # n_states, regime_stats, overlay_oos_sharpe,
                           # buyhold_oos_sharpe, sharpe_diff, jk_pvalue,
                           # deflated_sharpe, n_effective_trials, verdict, data_source
figures = assemble_regime_figures(result)   # {"regime_figure", "equity_figure"}
```

The `verdict` is a **pure function** of the OOS inference: it is structurally
unable to report `timing_edge` while Memmel-JK is insignificant or the Deflated
Sharpe (deflated by the full effective `n_trials` = `|n_states grid| × |feature
variants| × |cost grid|`) is non-positive.

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

## Validation

The hand-rolled kernel and inference are pinned against independent oracles and
property invariants (all green, coverage ≥ 85%, ruff + strict mypy clean):

| Check | Guarantee |
| --- | --- |
| HMM log-likelihood / smoothed posteriors / Viterbi path | match `hmmlearn.GaussianHMM` to `1e-6` on seeded 2/3-state data |
| Deflated Sharpe | matches the reused `dsr` reference to `1e-10` |
| Online filter | future-perturbation invariance / prefix-determinism (no lookahead) |
| EM | monotonic log-likelihood increase per iteration |
| Posteriors / transitions | rows sum to 1; transition rows stochastic |
| State labels | canonical (ascending-mean) ordering, relabeling-invariant across folds |
| Honest null | on the `regime_switch` fixture the overlay does **not** beat buy-and-hold → `no_timing_edge` |

## Limitations

- **Smoothed / Viterbi posteriors are non-tradable** — they peek ahead; only the
  online filter may drive an out-of-sample signal.
- **Survivorship bias is N/A** — the analysis runs on a single index series
  (synthetic or one ticker), not a cross-section selected on survival.
- **The timing overlay is the honest null**, not a product: its in-sample edge
  decays out-of-sample once the Deflated Sharpe with the full effective `n_trials`
  is applied. Regime *characterization* is the deliverable.

## References

Hamilton (1989); Ang & Bekaert (2002); Rabiner (1989); Bailey & López de Prado
(2014, Deflated Sharpe Ratio); Memmel (2003, Sharpe-difference test).

## License

MIT — see `LICENSE`.
