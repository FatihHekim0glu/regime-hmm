# ADR-0003: EM uses seeded multi-restart + a covariance floor

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** regime-hmm maintainers
- **Related:** [ADR-0002](0002-state-canonicalization.md) (canonicalization that absorbs restart label-shuffle), [ADR-0005](0005-hmmlearn-parity-oracle.md) (the oracle the kernel is pinned to)

## Context

Baum-Welch EM for a Gaussian HMM has two well-known failure modes:

1. **Local optima.** The log-likelihood surface is multimodal. A single EM run from
   one initialization regularly converges to a poor local optimum, and which one
   depends entirely on the (random) starting parameters. A benchmark whose fit
   quality is a coin-flip is not reproducible.
2. **Covariance collapse / singularity.** A state can collapse onto a handful of
   near-identical observations during EM. Its estimated variance then heads toward
   zero, the Gaussian emission log-density heads toward `+inf`, and the fit either
   diverges or produces a degenerate, non-invertible covariance. This is the
   classic unbounded-likelihood pathology of Gaussian mixtures/HMMs.

Both must be handled deterministically, because the test suite pins the fit against
an oracle and against golden regressions.

## Decision

**Seeded multiple restarts.** `fit_hmm` runs `n_restarts` independent EM
optimizations from distinct initializations and keeps the one with the highest
training log-likelihood. The restart seeds are reproducible PCG64 **substreams**
spawned from the master seed (`regimehmm._rng.spawn_substreams`), so a fixed master
seed reproduces the entire fit, including which restart wins, byte-for-byte.

**A strictly positive covariance floor.** Every M-step re-estimated covariance is
passed through `floor_covariance` (default `floor = 1e-6`): for `diag` covariances
each per-feature variance is raised to at least the floor; for `full` covariances
the floor is added to the diagonal so each matrix stays symmetric positive-definite.
This is the single chokepoint that keeps a collapsed state finite.

**A monotonicity assertion.** Within a single restart the per-iteration log-
likelihood is asserted non-decreasing (Baum-Welch is monotone in exact arithmetic),
with a tiny tolerance (`1e-6` nats) for floating-point round-off. A genuine
decrease is a bug and fails loudly.

## Consequences

- **Positive.** Fits are reproducible and resistant to local optima; the same seed
  always yields the same model, and the kept fit matches the `hmmlearn` oracle to
  `1e-6` ([ADR-0005](0005-hmmlearn-parity-oracle.md)).
- **Positive.** The covariance floor makes the emission model well-defined for every
  state on every iteration; a degenerate/singular-covariance unit test confirms a
  collapsed state stays finite instead of blowing up the likelihood.
- **Positive.** The monotonic-LL assertion is a cheap, always-on invariant that
  catches M-step bugs immediately.
- **Cost.** `n_restarts` multiplies fit time linearly. The default (8 for the
  library, fewer in the cheap request-time path) trades compute for reliability;
  it is a tunable, not a hidden constant.
- **Cost.** The floor introduces a small, known bias for genuinely low-variance
  states. `1e-6` on standardized features is negligible relative to the regime
  volatilities and is the same order as the oracle's `min_covar`, so parity holds.
