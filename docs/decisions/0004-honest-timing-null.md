# ADR-0004: The timing verdict is a pure function with the honest-null built in

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** regime-hmm maintainers
- **Related:** [ADR-0001](0001-online-filter-only-tradable.md) (the leakage-free OOS signal the verdict consumes), [ADR-0005](0005-hmmlearn-parity-oracle.md)

## Context

The honest finding of this project is that the regime-timing exposure overlay does
**not** reliably beat buy-and-hold out-of-sample after costs, and that whatever
in-sample edge appears **decays** once you correct for how many configurations were
tried. The temptation in any timing study is to narrate around a fragile result:
cherry-pick the cost level or feature set where the point Sharpe happens to land
above buy-and-hold, quote the raw Sharpe gap, and call it an edge.

Three statistical facts make that narration dishonest:

- a **point** Sharpe gap on one finite OOS realization is noise; it lands either
  side of zero by chance;
- the right test is a **Sharpe-difference** test (Memmel-corrected Jobson-Korkie),
  and on this data it is insignificant;
- searching a grid of `n_states` × feature sets × cost levels inflates the best
  observed Sharpe, the **Deflated Sharpe** (Bailey-López de Prado) corrects for
  exactly this selection, and the correct effective `n_trials` is the **size of the
  whole grid**, not 1.

## Decision

The headline verdict is a **pure function** of the OOS inference,
`derive_timing_verdict(jk_pvalue, deflated_sharpe, sharpe_diff)`, with the
honest-null discipline encoded in the control flow:

```
if  sharpe_diff <= 0            (the overlay does not even out-perform)
 or jk_pvalue   >= alpha        (the gap is statistically indistinguishable from 0)
 or deflated_sharpe <= 0:       (degenerate-deflation guard)
        -> NO_TIMING_EDGE
elif deflated_sharpe < dsr_threshold:   (significant gap, fails multiplicity)
        -> MARGINAL
else:                                   (significant AND survives deflation)
        -> TIMING_EDGE
```

It is therefore **structurally impossible** to emit `TIMING_EDGE` while the
Memmel-JK test is insignificant or the Deflated Sharpe is non-positive. The verdict
is *derived* from the evidence, never narrated.

The Deflated Sharpe is always deflated by the **full** effective
`n_trials = |n_states grid| × |feature variants| × |cost grid| = 3 × 3 × 4 = 36`,
counted by `effective_n_trials`, which raises if any factor is `< 1`. The
multiplicity can never be silently collapsed to 1.

## Consequences

- **Positive.** The README's headline cannot drift from the code: the only way to
  flip the verdict to `timing_edge` is for the overlay to *actually* beat
  buy-and-hold with a Sharpe gap that is both significant and survives 36-trial
  deflation. On the synthetic data it does not, so the verdict is `no_timing_edge`
  (regression-pinned), even in runs where the raw JK p-value is small, because
  there the Sharpe gap is *negative* (the overlay under-performs).
- **Positive.** The verdict truth table is unit-tested over the full
  `(jk_pvalue, deflated_sharpe, sharpe_diff)` quadrant, including NaN inputs (NaN
  fails its positivity check, collapsing to `no_timing_edge`).
- **Cost.** Every configuration axis the library exposes (states, features, costs)
  *correctly* enlarges `n_trials` and deflates harder. Exploring more is not free, which is the point. Adding an axis without updating the grids would understate the
  deflation, so the grids live next to the entrypoint and feed both the analysis and
  the count.
- **Honest by construction.** The project ships a null result and is *engineered* to
  be unable to oversell it.
