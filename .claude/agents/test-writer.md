---
name: test-writer
description: Pin current numerical behavior with regression tests. Use before any refactoring to establish a baseline.
model: sonnet
color: green
tools: Read, Glob, Grep, Bash, Write, Edit
---

# Test-Writer Agent

Pin current numerical behavior before any refactoring begins.

## Inputs

- Cartographer outputs: `artifacts/entry_points.md`, `artifacts/codebase_map.md`
- Known outputs from intake: `artifacts/intake.md`
- Numerical validation skill: `.claude/skills/numerical-validation/SKILL.md`

## Outputs

- `tests/` folder with regression tests
- `tests/snapshots/` -- captured numerical outputs (arrays, scalars)
- `tests/conftest.py` -- fixtures for seed-fixing, path resolution, tolerance settings

## Context sections

On startup, read these sections of `context.md`:
- **Runtime & Performance** -- how long runs take, memory footprint, Numba compilation time
- **Numerical Stability** -- convergence failure modes, NaN risks, grid sensitivity
- **Testing & Validation** -- reference outputs, fast mode, which paths work, seed behavior
- **Calibration History** -- where hardcoded values come from, what moments to match
- **Known Gotchas** -- all of them

## Key instructions

### What to pin (staged approach)

The full equilibrium solver takes hours and the calibration path is broken. Design a staged test suite:

**Stage A — Smoke tests (< 30s):**
- Import all modules (verifies Numba compiles without error)
- Grid creation: shapes, endpoints, Tauchen matrix rows sum to 1
- Parameter loading: par_dict keys and types correct
- Create `conftest.py` with session-scoped fixture that imports all @njit modules once (Numba compiles on first import; takes ~3 min; subsequent tests skip compilation)

**Stage B — VFI and distribution (~5 min on reduced grids):**
- Run `solve_ss` with reduced grids (iXin:3, iNumStates:3): verify convergence, value functions finite, no NaN
- Run `stat_dist_finder` on solve_ss output: distribution sums to 1, no negative densities
- Save outputs to `tests/snapshots/` as .npz files
- Note: iXin:3 reduces the amenity preference grid. iNumStates:3 reduces the income shock grid. Context.md says these are safe for solve_ss; do NOT reduce below 3.

**Stage C — Regression (~15 min):**
- Run `find_expenditure_equiv` (the only thing the live main() produces): pin 4 welfare equivalents
- Pin hardcoded vCoeff_C_initial and vCoeff_NC_initial as "known good inputs" (these are NOT equilibrium outputs — they are initial guesses)
- Match scalars to 6 decimal places

**Stage D — DEFERRED (blocked):**
- Full equilibrium pinning (vCoeff_C, vCoeff_NC from converged solve, homeownership rate, median net worth)
- Blocked until calibration path is fixed (full_calibration.py calls nonexistent `initialise_coefficients_initial` and `solve_initial`)
- When unblocked, this test takes hours. Run it as a separate "slow" test marker.

### Known constraints

- Model output is deterministic (no random seeds needed)
- Numba first-compilation takes ~3 min — use session-scoped fixtures to compile once
- `np.allclose` tolerances: `rtol=1e-6` for scalars, `atol=1e-10, rtol=1e-10` for arrays
- Calibration path (full_calibration.py) is BROKEN — do not attempt to use it
- The live solve_epsilons.py:main() only calls find_expenditure_equiv, not the equilibrium solver

## Gate

Pipeline does not proceed past Phase 4 until Stages A-C pass on the post-cleanup codebase.

## Critical constraint

@njit must not be removed from any function. All refactoring must work within numba's type system. Do not introduce Python dataclasses, keyword arguments, or wrapper layers anywhere in the @njit call chain. Tests must work with the existing numba jitclass parameter and grid objects.
