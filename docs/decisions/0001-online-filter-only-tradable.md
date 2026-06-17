# ADR-0001: Only the online forward filter is tradable; smoothed/Viterbi are EDA-only

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** regime-hmm maintainers
- **Related:** [ADR-0004](0004-honest-timing-null.md) (the verdict that consumes the OOS signal)

## Context

A Gaussian HMM yields three different "what regime is it" answers:

- the **online forward filter** posterior `p(state_t | x_1..x_t)`, the normalized
  forward message `alpha_t`, which conditions on data **up to `t` only**;
- the **smoothed** (forward-backward) posterior `gamma_t = p(state_t | x_1..x_T)`,
  which conditions on the **whole** sample, including the future;
- the **Viterbi** MAP path, the single most likely state *sequence* over the whole
  sample — also a function of the future.

The smoothed and Viterbi answers are far cleaner: they de-noise the regime path
using hindsight, so a backtest built on them looks spectacular. That is exactly the
trap. Any out-of-sample label or trading signal derived from a smoothed or Viterbi
posterior is **look-ahead leakage**: at decision time `t` you do not have
`x_{t+1}..x_T`, so you could not have known that label. This is the single most
common way a regime-timing backtest fools its author, and it is the specific risk
this project exists to *not* fall into.

## Decision

The **online forward filter is the ONLY tradable regime signal.** The overlay,
the OOS labels, and the verdict consume the filtered posterior exclusively. The
exposure decided at `t` is applied via `signal.shift(1)`.

The smoothed posterior and the Viterbi path are kept (they are genuinely useful for
in-sample exploration and for the regime-shaded figure) but are **EDA-only**: they
must never be turned into an OOS label or signal. The figure even shades by the
*filtered* labels, not the smoothed ones, so the picture a visitor sees is the
honest, no-lookahead one.

This is enforced structurally, not by convention:

- a Hypothesis **property test** perturbs the return series at indices `> t` and
  asserts the filtered posterior at `t` is bit-identical (future-perturbation
  invariance / prefix-determinism);
- the overlay's `shift(1)` chokepoint is regression-pinned (the first OOS bar is
  dropped because there is no prior decision).

## Consequences

- **Positive.** The OOS signal is incapable of look-ahead by construction, so the
  honest-null result cannot be an artifact of leakage. This is the whole credibility
  of the project.
- **Positive.** Keeping smoothed/Viterbi as first-class EDA lets a reader *see* how
  much cleaner the hindsight path is — which is precisely the intuition for why
  leaking it would inflate a backtest.
- **Cost.** The filtered path is noisier than the smoothed one, so the overlay
  trades more and characterization-from-filtered-labels is slightly less crisp than
  it would be from `gamma`. That is the correct, honest cost of not cheating.
- **Risk addressed.** "Smoothed-posterior leakage" — the project's top risk — is
  eliminated by making the filter the only signal and property-testing the
  no-lookahead invariant.
