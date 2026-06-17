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
(future-perturbation invariance / prefix-determinism). See
[ADR-0001](docs/decisions/0001-online-filter-only-tradable.md).

## What the synthetic experiment actually shows

The entire test suite runs offline on a seeded synthetic regime-switch generator
(`data.py`): a sticky two-state Markov chain (calm low-vol vs. turbulent high-vol)
emitting Gaussian returns, with high ground-truth persistence and sharply
different volatilities. The numbers below are produced by the
[Reproduce](#reproduce) commands (`--n-states 2 --feature-set returns --seed 7`)
and are deterministic for a fixed seed.

**1. Characterization — the win.** A 2-state fit recovers the two persistent
regimes cleanly (online-filtered, canonical order, annualized):

| Regime | Mean | Vol | Persistence | Expected duration | Frequency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` risk-off (high-vol) | −0.43 | 0.19 | 0.984 | ~62 d | 0.48 |
| `1` calm (low-vol)      | +0.16 | 0.10 | 0.986 | ~69 d | 0.53 |

Both regimes are highly persistent (~0.98, multi-month expected dwell times), the
risk-off regime carries roughly double the volatility and a sharply negative mean,
and the labels are stable across folds (canonical ascending-mean ordering,
[ADR-0002](docs/decisions/0002-state-canonicalization.md)).

**2. Timing — the honest null.** The regime-conditioned exposure overlay (cut
exposure in the risk-off regime, online filter + `shift(1)`, 10 bps per side)
does **not** beat buy-and-hold OOS. The headline run
(`n_states=2, feature_set="returns", cost_bps=10`):

| Metric | Value |
| --- | ---: |
| Overlay OOS Sharpe | −1.52 |
| Buy-and-hold OOS Sharpe | −0.83 |
| Sharpe gap (overlay − buy-hold) | −0.69 |
| Memmel-JK p-value | 0.001 |
| Deflated Sharpe (n_trials = 36) | 0.00 |
| Effective `n_trials` (3 × 3 × 4) | 36 |
| **Verdict** | **`no_timing_edge`** |

The gap is *negative* — the overlay loses — so even though the Memmel-JK test is
nominally significant, the verdict is structurally `no_timing_edge` (you cannot
claim a "timing edge" by *under*performing). Across the `n_states` × feature ×
cost grid the overlay's point Sharpe lands either side of buy-and-hold on any
single finite realization, but the **gap is never significantly positive** and the
Deflated Sharpe (deflated by the full 36-trial grid) never clears its threshold —
so the verdict is `no_timing_edge` everywhere. That is the honest finding the
literature predicts, mechanically enforced by the pure-function verdict
([ADR-0004](docs/decisions/0004-honest-timing-null.md)).

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

For how the layers fit together and the invariants they guarantee, see
[`docs/DESIGN.md`](docs/DESIGN.md) and the numbered ADRs in
[`docs/decisions/`](docs/decisions/).

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
is never imported by `src/` or the deployed API
([ADR-0005](docs/decisions/0005-hmmlearn-parity-oracle.md)).

## Reproduce

Every number above is regenerated by the test suite and the CLI from a fixed seed
(no network — the synthetic generator is the default data source):

```bash
# 1. Environment
uv venv && uv pip install -e '.[data,viz,dev]'

# 2. Quality gates (ruff + strict mypy + coverage >= 85)
uv run ruff check src
uv run mypy src
uv run pytest -q --cov=regimehmm --cov-report=term --cov-fail-under=85

# 3. The headline honest-null run on synthetic data (verdict: no_timing_edge)
uv run regime-hmm backtest --n-states 2 --feature-set returns --cost-bps 10 --seed 7

# 4. The characterization (per-regime mean/vol/persistence/duration)
uv run regime-hmm fit --n-states 2 --seed 7
uv run regime-hmm decode --n-states 2 --seed 7
```

The parity tests pin the kernel against `hmmlearn.GaussianHMM` to `1e-6`; the
honest-null regression pins the verdict to `no_timing_edge`; the online-filter
property tests pin no-lookahead. Same seed → identical outputs (the `RunManifest`
carries a BLAKE2b config hash).

## Validation

The hand-rolled kernel and inference are pinned against independent oracles and
property invariants. Each row is enforced by a test under `tests/` (all green,
coverage ≥ 85%, ruff + strict mypy clean):

| What | Oracle / invariant | Tolerance | Test |
| --- | --- | --- | --- |
| HMM emission log-density | `hmmlearn.GaussianHMM._compute_log_likelihood` | `1e-6` | `tests/parity/test_hmm_parity.py` |
| Smoothed `gamma` + total log-likelihood | `hmmlearn.GaussianHMM` (`score` / `predict_proba`) | `1e-6` | `tests/parity/test_hmm_parity.py` |
| Viterbi MAP path | `hmmlearn.GaussianHMM.decode` | exact labels | `tests/parity/test_hmm_parity.py` |
| Deflated Sharpe | the reused `dsr` reference | `1e-10` | `tests/parity/test_hmm_parity.py` |
| Online filter | future-perturbation invariance / prefix-determinism (no lookahead) | structural | `tests/property/test_hmm_properties.py` |
| EM | monotonic log-likelihood increase per iteration | `1e-6` nats | `tests/unit/test_hmm_kernel.py` |
| Posteriors / transitions | rows sum to 1; transition rows stochastic | `1e-9` | `tests/property/test_hmm_properties.py` |
| State labels | canonical (ascending-mean) ordering, relabeling-invariant across folds | structural | `tests/unit/test_canonicalize.py` |
| Covariance floor | a singular / collapsed state stays finite | `floor = 1e-6` | `tests/unit/test_hmm_kernel.py` |
| Honest null | overlay does **not** beat buy-and-hold → `no_timing_edge` | structural | `tests/regression/test_overlay_honest_null.py` |
| Verdict truth table | `timing_edge` impossible while JK insignificant or DSR ≤ 0 | structural | `tests/regression/test_verdict.py` |

`hmmlearn` is imported **only** by the parity tests, never by `src/` or the API.

## Limitations

- **Smoothed / Viterbi posteriors are non-tradable** — they condition on the whole
  sample and so peek ahead. Only the online filter may drive an out-of-sample
  signal; smoothed/Viterbi outputs are exploratory (EDA) only. This is the central
  design constraint, not an afterthought ([ADR-0001](docs/decisions/0001-online-filter-only-tradable.md)).
- **Survivorship bias is N/A** — the analysis runs on a single index series
  (synthetic, or one ticker such as SPY), not a cross-section selected on survival.
  There is no universe-construction step in which a survivorship screen could enter,
  so the usual cross-sectional survivorship caveat does not apply here.
- **The timing overlay is the honest null, not a product** — its in-sample edge
  decays out-of-sample once the Deflated Sharpe with the full effective `n_trials`
  (= 36 = 3 `n_states` × 3 feature sets × 4 cost levels) is applied. Regime
  *characterization* is the deliverable ([ADR-0004](docs/decisions/0004-honest-timing-null.md)).
- **Single-asset, Gaussian-emission HMM** — emissions are Gaussian over a small
  causal feature set; fat tails and asymmetric regime transitions beyond a
  first-order Markov chain are out of scope. The model is fit at request time on a
  small panel, not pre-trained.

## References

- Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle.* **Econometrica** 57(2),
  357–384. (Markov-switching model of the macro cycle.)
- Ang, A., & Bekaert, G. (2002). *International Asset Allocation with Regime
  Shifts.* **Review of Financial Studies** 15(4), 1137–1187. (Regimes in asset
  returns; timing is hard.)
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition.* **Proceedings of the IEEE** 77(2), 257–286.
  (Forward-backward, Baum-Welch, Viterbi — the algorithms implemented here.)
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.*
  **The Journal of Portfolio Management** 40(5), 94–107. (The multiplicity
  correction the verdict applies.)
- Memmel, C. (2003). *Performance Hypothesis Testing with the Sharpe Ratio.*
  **Finance Letters** 1, 21–23. (Corrected Jobson-Korkie Sharpe-difference test.)

## License

MIT — see `LICENSE`.
