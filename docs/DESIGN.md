# Design

This document explains how `regime-hmm` is put together: the layering, the data
flow through one request, the leakage discipline that is the whole point of the
project, the invariants the compute core guarantees, and the testing strategy that
keeps the honest headline honest. For *why* individual contested choices were made,
see the numbered ADRs in [`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI,
  network, or the `hmmlearn` oracle along.
- A faithful from-scratch Gaussian HMM (emission log-density, log-space
  forward-backward, Baum-Welch EM, Viterbi, online filter) parity-tested to `1e-6`
  against `hmmlearn`.
- An out-of-sample regime signal that is **structurally** incapable of look-ahead:
  the only tradable posterior is the online forward filter (data ≤ t).
- A statistically defensible timing verdict that survives multiplicity correction
  and is *mechanically* prevented from over-claiming.

**Non-goals**

- Beating buy-and-hold. The honest finding is that the regime-timing overlay does
  not, after costs, by a significant margin — and its in-sample edge decays under
  the Deflated Sharpe.
- A live trading system. This is a research/characterization library.
- A general time-series toolkit. The HMM exists to characterize market regimes.

## The leakage rule (read this first)

The single design constraint everything else serves: **the only tradable regime
signal is the online forward filter posterior** `p(state_t | x_1..x_t)`, which
conditions on data **up to `t` only**.

- The **smoothed** (forward-backward) posterior `gamma` and the **Viterbi** MAP
  path both condition on the *whole* sample. They peek ahead. They are in-sample
  **EDA only** and are never turned into an out-of-sample label or signal.
- The overlay consumes the filtered posterior and applies the exposure decided at
  `t` via `signal.shift(1)`, so the return realized at `t` is earned with an
  exposure chosen strictly before `t`.

This is enforced, not merely documented: a property test perturbs returns *after*
`t` and asserts the filtered posterior at `t` is unchanged (future-perturbation
invariance / prefix-determinism). See
[ADR-0001](decisions/0001-online-filter-only-tradable.md).

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/regimehmm/` has **zero import-time side effects** — every heavy/lazy
dependency is imported inside functions, and no fit, network call, or RNG draw runs
at import (subprocess import-purity tested).

```
                 cli.py (Typer)        plots.py (Plotly)        analysis.py
                      |                      |                       |
   ┌──────────────────┴──────────────────────┴───────────────────────┘
   │                          evaluation/
   │           dsr.py · comparison.py · verdict.py
   │   (Deflated Sharpe, Memmel-JK + block bootstrap, pure timing verdict)
   ├──────────────────────────────────────────────────────────────────
   │            backtest/                        regimes/
   │   walk_forward · costs · stats     canonicalize · characterize
   │   overlay (filtered exposure)      (stable labels, per-regime stats)
   ├──────────────────────────────────────────────────────────────────
   │                          hmm/
   │   kernel · forward_backward · em · viterbi · filter (+ HMMModel)
   │   (Gaussian emissions, log-space FB, Baum-Welch, MAP, ONLINE FILTER)
   ├──────────────────────────────────────────────────────────────────
   │   data.py · data_providers/        foundation (no internal deps)
   │   (synthetic generator, Polygon    _validation · _constants · _typing
   │    EOD → synthetic fallback)        _exceptions · _manifest · _rng
   └──────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

Reused, renamed-from-`hrp` helpers. `_constants.py` (`PERIODS_PER_YEAR = 252`,
`EPS`), `_validation.py` (shape/finiteness/length guards), `_typing.py` /
`_exceptions.py` (aliases + exception taxonomy), and `_manifest.py` / `_rng.py`
(`RunManifest` with a BLAKE2b config hash plus seeded PCG64 substreams). The same
master seed reproduces a fit byte-for-byte.

### `hmm/`

The only genuinely new code. `kernel.py` computes the Gaussian emission
log-density for `diag` and `full` covariances and applies the covariance **floor**
that keeps a collapsed state finite ([ADR-0003](decisions/0003-em-restarts-covariance-floor.md)).
`forward_backward.py` runs the E-step in log space (`alpha`, `beta`, smoothed
`gamma`, pair-marginals `xi`, log-likelihood). `em.py` is Baum-Welch with multiple
seeded restarts, keeping the highest-likelihood fit and asserting per-iteration
monotonic log-likelihood. `viterbi.py` recovers the MAP path (EDA-only).
`filter.py` holds the frozen `HMMModel` and the **online forward filter** — the
only tradable posterior.

### `regimes/`

`canonicalize.py` sorts states by ascending mean conditional return (vol
tie-break) and returns a permutation that relabels the model and any decoded
series, so "regime 0" means the same thing across folds
([ADR-0002](decisions/0002-state-canonicalization.md)). `characterize.py` produces
per-regime mean / vol / persistence / expected duration / max-drawdown.

### `backtest/`

Reused `walk_forward.py` (anchored windows, purge + embargo, `signal.shift(1)`),
`costs.py` (per-side bps), and `stats.py` (Sharpe/vol/turnover/drawdown). The new
`overlay.py` builds the regime-conditioned exposure from the filtered posterior,
charges per-side costs on exposure changes, sweeps a cost grid, and scores
overlay-vs-buy-and-hold on the **identical** OOS index.

### `evaluation/`

Reused `dsr.py` (Probabilistic + Deflated Sharpe with the full-grid `n_trials`) and
`comparison.py` (Memmel-corrected Jobson-Korkie Sharpe-difference + stationary
block bootstrap). The new `verdict.py` is a **pure function** mapping
`(jk_pvalue, deflated_sharpe, sharpe_diff)` to the verdict enum
([ADR-0004](decisions/0004-honest-timing-null.md)).

## Data flow through one request

```
returns (provided)  ──►  ── OR ──  ticker ──► Polygon EOD ──► (fail) ──► synthetic
        │                                          │                        │
        └──────────────────────┬───────────────────┴────────────────────────┘
                               ▼
           causal features (trailing windows) + train-only standardization
                               │
                               ▼
        fit_hmm (Baum-Welch, seeded restarts) ─► canonicalize_model
                               │
                               ▼  ONLINE FILTER (data ≤ t)   ◄── the only tradable signal
        filtered posterior ────┼──► hard labels ─► characterize_regimes
                               │
                               ▼  regime_exposure ─► shift(1) ─► per-side bps cost
        overlay vs buy-and-hold (identical OOS index, after costs)
                               │
                               ▼
   Memmel-JK Sharpe-diff p-value  +  Deflated Sharpe (n_trials = 3×3×4 = 36)
                               │
                               ▼
              verdict.py  ──►  no_timing_edge | marginal | timing_edge
```

The deployed backend calls exactly one function — `run_regime_analysis(...)` — and
fits the HMM at request time on a small panel (cheap; no pre-trained artifact).
Polygon failure degrades to the seeded synthetic generator so the call never
hard-fails; `data_source` reports `polygon` / `synthetic` / `provided`.

## Key invariants

The compute core guarantees, and tests enforce:

1. **No look-ahead.** The filtered posterior at `t` is invariant to perturbing
   returns after `t` (future-perturbation invariance / prefix-determinism). The
   overlay exposure is applied via `shift(1)`.
2. **Smoothed/Viterbi are non-tradable.** They condition on the whole sample; they
   feed EDA and figures only, never an OOS label.
3. **Stochastic rows.** Posterior rows sum to 1; the transition matrix rows are
   stochastic.
4. **Canonical labels.** States are ordered by ascending mean return; the
   characterization is relabeling-invariant across folds.
5. **EM monotonicity.** The per-iteration training log-likelihood is non-decreasing
   within a restart (Baum-Welch is monotone), asserted at fit time.
6. **Covariance floor.** A degenerate/collapsed state stays positive-definite and
   finite (`floor = 1e-6`) — EM cannot emit a singular state.
7. **Honest `n_trials`.** The Deflated Sharpe is deflated by the full grid product
   (`|n_states| × |feature variants| × |cost grid|` = 36), never collapsed to 1.
8. **Verdict safety.** `derive_timing_verdict` cannot emit `timing_edge` while
   Memmel-JK is insignificant or the Deflated Sharpe ≤ 0 (truth-table unit-tested).
9. **Determinism.** Same seed → byte-identical outputs (the `RunManifest` carries a
   BLAKE2b config hash).
10. **Import purity.** Importing any `src/regimehmm` module triggers no I/O, no
    network, no RNG draw, no fit (subprocess-tested). `hmmlearn` is never imported.

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`):

- **`unit/`** — isolated kernels: the emission log-density + covariance floor, the
  canonical ordering, the characterization, the verdict truth table.
- **`property/`** (Hypothesis) — the invariants above: online-filter no-lookahead
  (future-perturbation invariance / prefix-determinism / shift-equivariance),
  posterior/transition stochasticity, canonical ordering, feature scale-invariance.
- **`parity/`** — golden checks against independent references: HMM log-likelihood,
  smoothed `gamma`, and Viterbi path vs `hmmlearn.GaussianHMM` to `1e-6` on seeded
  2- and 3-state data; Deflated Sharpe vs the reused `dsr` reference to `1e-10`.
- **`regression/`** — the honest null, locked: on `regime_switch` the overlay does
  not beat buy-and-hold and Memmel-JK is insignificant → `no_timing_edge`; the
  cost-grid is monotone; the verdict truth table; no-lookahead backtest regression.
- **`integration/`** — the full `run_regime_analysis` entrypoint and the Typer CLI
  end-to-end on the synthetic fixture (offline; effective `n_trials` = 36).

Seeded fixtures in `conftest.py` (`one_factor`, `regime_switch`, `pure_noise`) give
every layer deterministic inputs with known structure. The whole suite runs
**offline** — no test touches the network.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`regime-hmm[data]` (numpy/pandas/scipy/sklearn — **not** `hmmlearn`, **not**
`[viz]`/`[dev]`) under `api/lib/regime_hmm/` and exposes
`POST /tools/regime-hmm/run`, fitting at request time and returning summary scalars
(all `_safe_float`-cleaned) plus Plotly `{data, layout}` figures. Polygon failure
falls back to synthetic; the response never hard-fails. The frontend renders the two
figures and surfaces the pure-derived `verdict` and the honest caption — regime
characterization is the win, timing is not free money — as the first thing a
visitor reads.
