---
name: macro-model-architecture
description: >
  Reference guide for our life-cycle macro model codebase solved via the
  endogenous grid method (EGM). Use when reading, navigating, or modifying
  model code. Use when user mentions "model code", "our model", "the solver",
  grid construction, state variables, or any project file names.
metadata:
  author: TODO
  version: 0.1.0
---

# Macro Model Architecture Reference

Claude should consult this skill whenever working with the project codebase.
It encodes the ground-truth about file layout, conventions, and known issues.

---

## Model Overview

<!-- TODO: 1-2 paragraph summary of what the model does economically.
     Example: "Heterogeneous-agent life-cycle model with X choice margins,
     calibrated to Y, used for Z analysis." -->

**Inspiration / closest published model:** Kaplan life-cycle model (EGM solution)

<!-- TODO: link to paper or repo if public -->

### State Variables

<!-- TODO: fill in your actual states -->

| Symbol | Name | Grid points | Domain | Notes |
|--------|------|-------------|--------|-------|
| `a`    | assets | TODO | [a_min, a_max] | endogenous grid via EGM |
| TODO   | TODO | TODO | TODO | TODO |

### Choice Variables

<!-- TODO: list choice/control variables -->

- `c` — consumption (recovered from Euler equation via EGM)
- TODO

### Key Parameters

<!-- TODO: list the parameters that matter most for debugging/review -->

| Parameter | Variable name in code | Typical value | Description |
|-----------|----------------------|---------------|-------------|
| TODO | TODO | TODO | TODO |

---

## Codebase Layout

<!-- TODO: fill in your actual file tree and describe each file's role -->

```
project_root/
├── TODO_main_entry_point.py   # top-level solver / runner
├── TODO_model_params.py       # parameter definitions & calibration
├── TODO_grids.py              # grid construction
├── TODO_egm_step.py           # single EGM backward-induction step
├── TODO_simulation.py         # forward simulation on solved policy
├── TODO_utils.py              # interpolation, helpers
└── TODO_plots.py              # diagnostics & figures
```

**Entry point:** `TODO` — describe how a user runs the model end-to-end.

---

## Naming Conventions

<!-- TODO: document the conventions actually used (even if inconsistent —
     noting inconsistencies is valuable for the cleanup skill) -->

- Variables: TODO (e.g. snake_case everywhere? mixed?)
- Grids: TODO (e.g. `a_grid`, `grid_a`, `agrid`?)
- Policy functions: TODO
- Value functions: TODO
- Time/age indexing: TODO (0-indexed? 1-indexed? age or period?)

---

## Solution Method Details

### EGM Implementation

<!-- TODO: describe the specific EGM variant used -->

1. **Backward induction** from terminal period `T = TODO`
2. At each age, for each exogenous state:
   - TODO: describe the Euler-equation inversion step
   - TODO: describe how the endogenous grid is recovered
   - TODO: describe interpolation onto common grid (if applicable)
3. **Boundary / constraint handling:**
   - Borrowing constraint: TODO
   - Upper-grid extrapolation: TODO

### Convergence / Tolerance

<!-- TODO: if there's a stationary component or iteration, describe criteria -->

- TODO

---

## Known Technical Debt & Quirks

<!-- TODO: this is the most valuable section for the cleanup agent.
     List anything you already know is messy, duplicated, or fragile. -->

1. TODO — e.g. "grid construction is copy-pasted in 3 places"
2. TODO — e.g. "simulation uses a different interpolation method than solver"
3. TODO — e.g. "some parameters are hardcoded in function bodies"
4. TODO — e.g. "unused legacy code from earlier model version still present"

---

## How to Run

```bash
# TODO: minimal instructions to run the model
# e.g. python main.py --config baseline.yaml
```

**Expected runtime:** TODO
**Expected output:** TODO (e.g. policy functions saved to X, figures in Y)
