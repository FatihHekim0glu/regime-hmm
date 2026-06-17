# ADR-0002: Canonicalize states by ascending mean return (vol tie-break)

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** regime-hmm maintainers
- **Related:** [ADR-0003](0003-em-restarts-covariance-floor.md) (the restarts that make labels arbitrary)

## Context

A raw HMM fit assigns **arbitrary integer indices** to its hidden states. Which
physical regime ends up labelled `0` versus `2` is an accident of restart
initialization: the "high-vol crisis" regime might be state 2 in one fold and state
0 in the next. EM with seeded restarts ([ADR-0003](0003-em-restarts-covariance-floor.md))
makes this worse, because the kept restart — and therefore the label assignment —
can differ across folds even for the same data.

That arbitrariness is poison for everything downstream:

- per-regime characterization tables would shuffle their rows fold to fold;
- a golden regression on "regime 0's mean return" would be meaningless;
- the overlay's notion of "the risk-off state" would not be stable.

We need a deterministic, content-based labelling so that "regime 0" always denotes
the same *kind* of regime.

## Decision

After every fit, **canonicalize** the model: sort states by **ascending mean
conditional return**, tie-broken by **ascending volatility**, and apply the
resulting permutation to the model parameters (`pi`, the transition matrix, the
means, the covariances) and to any decoded state series.

The convention this fixes:

- **regime `0` is always the lowest-mean (most risk-off) regime;**
- the **highest-index** state (`n_states - 1`) is the risk-off / crisis regime the
  overlay cuts exposure in.

Canonicalization is a relabelling only — it is a permutation, so it changes no
likelihood, no posterior, and no filtered path; it just renames states.

## Consequences

- **Positive.** Characterization is **relabeling-invariant** across folds and runs:
  a property test fits, permutes the states, re-canonicalizes, and asserts the
  per-regime stats are identical. Golden regressions on per-regime numbers become
  meaningful.
- **Positive.** The overlay can name the risk-off state structurally
  (`n_states - 1`) instead of re-deriving it from a look-ahead label — so the
  "which state is risky" decision is itself leakage-free.
- **Positive.** Serialized models are comparable across runs because the label
  ordering is deterministic.
- **Cost.** Ties (two states with near-identical mean *and* vol) are resolved by a
  fixed rule rather than meaning; in a genuinely 1-regime series the ordering is
  arbitrary-but-deterministic, which is acceptable because there is no real regime
  structure to mislabel.
- **Convention chosen:** ascending mean (not descending, not by vol-first) so that
  the natural reading "regime 0 = worst returns" matches how a risk-off overlay
  thinks. Volatility is the tie-break because the defining feature of the regimes is
  their sharply different volatility.
