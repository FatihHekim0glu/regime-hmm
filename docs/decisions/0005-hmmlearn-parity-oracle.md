# ADR-0005: hmmlearn is a dev-only parity oracle, not a runtime dependency

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** regime-hmm maintainers
- **Related:** [ADR-0003](0003-em-restarts-covariance-floor.md) (the kernel being pinned)

## Context

The HMM kernel, emission log-density, log-space forward-backward, Baum-Welch
M-step, Viterbi, and the online filter, is hand-rolled in pure numpy/scipy. Hand-
rolled numerical code is exactly where silent bugs hide: an off-by-one in the
`xi` pair-marginals, a missing `logsumexp` normalization, a transposed transition
matrix. Reasoning alone is not enough; the kernel needs to be pinned against an
independent, trusted implementation.

`hmmlearn.GaussianHMM` is the obvious oracle: a mature, widely used Gaussian HMM.
The tempting-but-wrong move is to simply *depend on* `hmmlearn` for the production
fit. That would defeat the purpose of the project (writing the algorithm from
scratch), and it would also drag a heavier dependency into the deployed container
for no runtime benefit.

There is also a subtle correctness reason not to ship it: if the production code and
the test oracle are the same library, the parity test is vacuous, it can only ever
confirm "the code equals itself."

## Decision

`hmmlearn` is a **dev-only parity oracle**. It lives in the `[dev]` extra and is
imported **only** by `tests/parity/`. The production kernel is pure numpy/scipy and
**never** imports `hmmlearn`; neither does the vendored backend.

The parity tests pin, on seeded 2- and 3-state synthetic data:

- the emission log-density vs `hmmlearn`'s `_compute_log_likelihood` to `1e-6`;
- the smoothed `gamma` and total log-likelihood vs `hmmlearn` (`predict_proba` /
  `score`) to `1e-6`;
- the Viterbi MAP path vs `hmmlearn.decode` (exact labels).

Dependency hygiene is enforced:

- `pyproject.toml` keeps `hmmlearn` out of `[data]` (the runtime extra the backend
  vendors) and out of `[viz]`;
- the backend vendors `regime-hmm[data]` only, so `hmmlearn` is not installable in
  the API image;
- the import-purity test confirms `src/` imports trigger no `hmmlearn` import.

## Consequences

- **Positive.** The from-scratch kernel is verified against an independent reference
  to `1e-6`, so a regression in the hand-rolled math fails the parity suite
  immediately, and the oracle is genuinely independent of the production code, so
  the test has teeth.
- **Positive.** The runtime/container footprint stays lean (numpy/pandas/scipy/
  sklearn); the deployed tool fits at request time without `hmmlearn`.
- **Cost.** The hand-rolled kernel must keep matching `hmmlearn`'s conventions
  (e.g. `min_covar` ↔ the covariance floor) closely enough to stay within `1e-6`;
  parameter choices in [ADR-0003](0003-em-restarts-covariance-floor.md) are made
  with that parity in mind.
- **Risk addressed.** "We reimplemented the algorithm and it is subtly wrong" is
  caught by an independent oracle; "the oracle leaked into production" is caught by
  the dependency split and the import-purity test.
