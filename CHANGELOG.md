# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Out-of-sample leakage in the reported overlay Sharpe (correctness blocker).**
  `run_regime_analysis` (and the CLI `backtest`) previously fit the StandardScaler
  and HMM on the FULL feature window and then reported the resulting in-sample
  overlay Sharpe as `overlay_oos_sharpe` / `buyhold_oos_sharpe`. The reported OOS
  numbers now come from a new genuinely-out-of-sample anchored walk-forward,
  `walk_forward_regime_overlay`: per fold the scaler + HMM are refit on the TRAIN
  window only, the OOS regime labels are derived from the ONLINE FILTER only
  (never smoothed/Viterbi), `signal.shift(1)` is applied, and the overlay and
  buy-and-hold are scored on the IDENTICAL post-purge/embargo OOS index. The
  full-window fit is retained ONLY for the descriptive in-sample regime figure /
  characterization table. API response field names are unchanged.
- **Risk-off state selection.** The overlay now cuts exposure in the
  HIGHEST-VOLATILITY regime, selected from the per-regime characterization via
  `select_risk_off_state` (`argmax` of conditional vol). Previously it picked the
  last canonical state positionally, which after canonicalization is the
  highest-MEAN-return regime, generally not the highest-vol one.
- **Deflated Sharpe degeneracy.** `deflated_sharpe_ratio` is no longer called with
  `variance_of_trial_sharpes=0.0` (which collapses the expected-maximum benchmark).
  A real, non-degenerate cross-trial Sharpe variance is computed from the swept
  cost grid on the genuinely-OOS overlay stream; the effective `n_trials` remains
  the full `|n_states| x |feature| x |cost|` grid product.

### Added

- `walk_forward_regime_overlay` + `WalkForwardRegimeResult` and
  `select_risk_off_state` (public API). `RegimeAnalysisResult` gains `oos_states`
  (the per-fold no-lookahead OOS regime labels).
- End-to-end no-lookahead regression test: future-perturbing post-cutoff returns
  leaves the OOS overlay returns and per-fold OOS regime labels on the pre-cutoff
  portion unchanged, with a leaky full-sample-fit sensitivity arm that must trip
  the invariant (proving the test is not a tautology).

### Changed

- CI: the strict mypy step is now a hard gate (`continue-on-error` removed).
- README: documents the display-vs-OOS split, the max-vol risk-off selection, and
  a forward-looking survivorship note (any future cross-sectional stock overlay
  MUST route its universe through the point-in-time `sp500_universe` builder).

## [0.1.0] - 2026-06-17

Initial release: a complete, import-pure, typed Gaussian HMM for honest
market-regime characterization. Regime characterization is the deliverable; the
regime-timing overlay does **not** reliably beat buy-and-hold out-of-sample after
costs, and its in-sample edge decays under the Deflated Sharpe. 258 tests pass;
coverage 88% (gate ≥ 85%); ruff + strict mypy clean.

### Added

- **From-scratch Gaussian HMM** (`hmm/`): `kernel` (Gaussian emission log-density,
  diag/full covariance with a strictly positive floor), `forward_backward`
  (log-space alpha/beta, smoothed `gamma`, pair-marginals `xi`, log-likelihood),
  `em` (Baum-Welch with seeded PCG64-substream restarts and a monotonic-LL
  assertion), `viterbi` (MAP path, EDA-only), and `filter`, the **online forward
  filter**, the only tradable no-lookahead regime posterior, plus the frozen
  `HMMModel`. Parity-tested to `1e-6` against `hmmlearn.GaussianHMM`.
- **Regime layer** (`regimes/`): `canonicalize` (states sorted by ascending mean
  return, vol tie-break, for stable cross-fold labels) and `characterize`
  (per-regime mean / vol / persistence / expected duration / max-drawdown).
- **Backtest** (`backtest/`): the reused no-lookahead walk-forward engine, per-side
  bps `costs`, Sharpe/vol/turnover/drawdown `stats`, and the regime `overlay`
  (filtered-regime exposure, `signal.shift(1)` chokepoint, cost sensitivity grid)
  scored vs buy-and-hold on an identical OOS index.
- **Evaluation** (`evaluation/`): Probabilistic + Deflated Sharpe (`dsr`),
  Memmel-corrected Jobson-Korkie Sharpe-difference + stationary block bootstrap
  (`comparison`), and the pure-function `verdict` (`no_timing_edge` / `marginal` /
  `timing_edge`) with the honest-null discipline and the full effective
  `n_trials` = `|n_states grid| × |feature variants| × |cost grid|` = 36.
- **`analysis.run_regime_analysis(...)`**, the single public end-to-end entrypoint
  the backend calls: load returns → causal features + train-only standardization →
  fit → canonicalize → online-filter decode → characterize → overlay-vs-buy-and-hold
  after costs → Memmel-JK + Deflated Sharpe → structurally-constrained verdict.
  Returns a frozen, JSON-serializable `RegimeAnalysisResult`.
- **`analysis.assemble_regime_figures(...)`**, builds the two frontend Plotly
  `{data, layout}` figures (regime-shaded cumulative return by filtered labels; OOS
  equity overlay vs buy-and-hold). Both exported from the package top level.
- **`data.py`**, a seeded synthetic regime-switch generator (the entire offline
  test suite runs on it) plus a Polygon-EOD loader with graceful synthetic
  fallback; `plots.py`, lazy Plotly figure builders; `cli.py`, a Typer
  `fit` / `decode` / `backtest` CLI.
- **Reused, renamed-from-`hrp` infra**: `_constants`, `_typing`, `_exceptions`,
  `_validation`, `_manifest` (`RunManifest` with a BLAKE2b config hash), `_rng`
  (seeded PCG64 generator + substream spawning), the Polygon EOD provider, and the
  `dsr` / `comparison` / `walk_forward` / `costs` / `stats` modules.
- **Tests**: partitioned `unit` / `parity` / `property` / `regression` /
  `integration` suites with seeded `conftest` fixtures (`one_factor`,
  `regime_switch`, `pure_noise`), HMM parity to `1e-6`, online-filter
  no-lookahead (future-perturbation invariance / prefix-determinism), canonical
  ordering, stochastic transitions, EM monotonic-LL, the covariance-floor guard,
  the honest-null `no_timing_edge` regression, and the verdict truth table.
- **Docs**: `README` (honest headline + actual synthetic numbers + validation table
  + reproduce block + limitations + references), `docs/DESIGN.md`, and ADRs
  `0001` to `0005` (online-filter-only-tradable, state-canonicalization, EM
  restarts + covariance floor, honest-null verdict, hmmlearn-as-parity-oracle);
  `CITATION.cff`; MIT `LICENSE`; `CONTRIBUTING`.
- **CI**: `ci.yml` (ruff + strict mypy + pytest with `fail_under = 85`) and a
  `no-ai-attribution` guard that rejects AI co-author / "Generated with" trailers.

[0.1.0]: https://github.com/FatihHekim0glu/regime-hmm/releases/tag/v0.1.0
