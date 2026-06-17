# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Full implementations for the genuinely-new modules (HMM `kernel` /
  `forward_backward` / `em` / `viterbi` / `filter`, `regimes` canonicalize +
  characterize, the `overlay`, the `verdict`, `data`, `plots`, and the `cli`),
  wired into one coherent, green library (ruff + strict mypy clean, coverage ≥ 85%).
- `analysis.run_regime_analysis(...)` — the single public end-to-end entrypoint
  the backend calls: fit → canonicalize → online-filter decode → characterize →
  regime-timing overlay vs buy-and-hold (after costs) → Memmel-JK + Deflated Sharpe
  (full effective `n_trials`) → honest, structurally-constrained verdict. Returns a
  JSON-serializable `RegimeAnalysisResult` (`summary` + `meta`).
- `analysis.assemble_regime_figures(...)` — builds the regime-shaded and OOS-equity
  Plotly figures the frontend renders. Both exported from the package top level.
- Integration tests running the full entrypoint on the synthetic `regime_switch`
  fixture (honest-null `no_timing_edge`, full `n_effective_trials` = 36) and unit
  tests for the block-bootstrap Sharpe-gap CI.

## [0.1.0] - 2026-06-17

### Added

- Initial package skeleton (src-layout, import name `regimehmm`).
- Reused core helpers (renamed from the `hrp-portfolio` infra): `_constants`,
  `_typing`, `_exceptions`, `_validation`, `_manifest` (`RunManifest` with
  BLAKE2b config-hash), and `_rng` (seeded PCG64 generator + substream spawning).
- Reused evaluation/backtest infra: Probabilistic + Deflated Sharpe (`dsr`),
  Jobson-Korkie-Memmel + stationary block bootstrap (`comparison`), the
  no-lookahead walk-forward engine (`walk_forward`), per-side bps costs (`costs`),
  Sharpe/vol/turnover/drawdown stats (`stats`), and the Polygon EOD provider.
- Stub signatures with full contracts for the genuinely new modules:
  - `hmm/` — `kernel` (Gaussian emission log-density + covariance floor),
    `forward_backward` (log-space alpha/beta, smoothed `gamma`/`xi`),
    `em` (Baum-Welch with seeded restarts + monotonic-LL invariant),
    `viterbi` (MAP path, EDA-only), and `filter` (the ONLINE forward filter —
    the only tradable, no-lookahead regime posterior — plus the `HMMModel`).
  - `regimes/` — `canonicalize` (stable cross-fold state ordering) and
    `characterize` (per-regime mean/vol/persistence/duration/drawdown).
  - `backtest/overlay` — regime-conditioned exposure overlay vs buy-and-hold.
  - `evaluation/verdict` — pure-function regime-timing verdict with the honest-null
    discipline and the effective-`n_trials` count.
  - `data` (synthetic regime-switch generator + Polygon/synthetic loader),
    `plots` (lazy Plotly regime-shaded + OOS equity figures), and `cli` (Typer).
- Curated top-level `__init__.py` re-exporting the public API.
- Partitioned `tests/` (unit/parity/property/regression/integration) with seeded
  conftest fixtures (`one_factor`, `regime_switch`, `pure_noise`) and import-surface
  smoke tests.

[Unreleased]: https://github.com/FatihHekim0glu/regime-hmm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/regime-hmm/releases/tag/v0.1.0
