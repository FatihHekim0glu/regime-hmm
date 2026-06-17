# Contributing

Thanks for your interest in `regime-hmm`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with the data, viz, and dev extras.
uv venv
uv pip install -e '.[data,viz,dev]'
```

Prefix commands with `uv run` to use that env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src                                                   # lint
uv run mypy src                                                         # types (strict)
uv run pytest -q --cov=regimehmm --cov-report=term --cov-fail-under=85  # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) must pass on `src/regimehmm`.
- **Tests** (`pytest`) must pass with **coverage ≥ 85%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Design invariants (do not break these)

- **Import purity.** `src/regimehmm/` has ZERO import-time side effects: no
  network, no model fitting, no heavy-dependency imports at module scope. Heavy
  deps (Polygon/httpx, scikit-learn, Plotly, Typer, hmmlearn) are imported lazily
  inside the functions that use them.
- **No smoothed-posterior leakage.** The ONLY tradable regime signal is the ONLINE
  forward filter (`regimehmm.hmm.filter`), whose posterior at `t` uses data `<= t`
  only. The smoothed (forward-backward) and Viterbi posteriors peek ahead and are
  in-sample EDA ONLY — never an out-of-sample label or signal.
- **`hmmlearn` is a dev-only parity oracle.** It is in the `dev` extra and the test
  suite, never in `data` and never imported by `src/` or the API container. The
  production kernel is pure numpy/scipy.
- **Honest-null verdict.** The timing verdict is a pure function of the OOS
  inference and is structurally unable to claim the overlay beats buy-and-hold when
  Memmel-JK is insignificant or the Deflated Sharpe is non-positive.

## Commit hygiene

- Use clear, present-tense commit messages.
- **Do not** add AI-attribution trailers — no `Co-Authored-By: Claude`,
  no "Generated with Claude", no robot-emoji attribution lines. The
  `.github/workflows/no-ai-attribution.yml` guard fails any PR that contains them.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
