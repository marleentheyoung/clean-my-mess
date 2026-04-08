---
name: macro-model-review
description: >
  Code review patterns for numerical life-cycle models solved with EGM.
  Use when diagnosing bugs, reviewing code quality, or identifying
  performance issues in macro model code. Use when user says "review",
  "diagnose", "what's wrong with", "check the code", "find issues",
  or "code smells" in context of the model.
metadata:
  author: TODO
  version: 0.1.0
---

# Macro Model Code Review Patterns

Checklist-driven review patterns for life-cycle models with EGM solutions.
Claude should work through relevant sections when reviewing model code.

---

## 1. Grid Construction Issues

### Common problems
- **Non-monotonic grids** after transformations (log-spacing then shifting)
- **Insufficient points** in regions of high curvature (near borrowing constraint)
- **Hardcoded grid bounds** that don't adapt to parameterisation
- **Duplicate grid construction** in multiple files (drift risk)

### What to check
```python
# Grid should be strictly increasing
assert np.all(np.diff(a_grid) > 0), "Grid not monotonic"

# Check for adequate density near constraint
lower_10pct = a_grid[a_grid < a_grid[0] + 0.1 * (a_grid[-1] - a_grid[0])]
# Should have meaningful fraction of points here
```

### EGM-specific grid issues
- Endogenous grid points can **cluster or cross** — must detect & handle
- After EGM step, check that resulting consumption is positive everywhere
- If using upper-envelope method for non-convexities: verify envelope is correct

---

## 2. Interpolation & Extrapolation

### Common problems
- **Linear interp where shape matters** — cubic near kinks, linear elsewhere
- **Extrapolation beyond grid** silently returning garbage
- **Inconsistent interpolation** between solver and simulator
- **Repeated scipy.interpolate object creation** inside tight loops

### What to check
- What happens when query point < grid_min or > grid_max?
- Is the same interpolation method used in backward induction AND simulation?
- Are interpolation objects created once and reused, or rebuilt per call?

---

## 3. Euler Equation / FOC Inversion

### Common problems
- **Numerical inversion errors** when marginal utility is near-flat
- **Wrong sign or off-by-one** in discount factor application
- **Missing or incorrect survival probabilities** in life-cycle discounting
- **Taste shifters / preference parameters** applied inconsistently

### What to check
```python
# After EGM inversion, verify Euler equation holds approximately
# (useful diagnostic even if not used at runtime)
euler_residual = u_prime(c_t) - beta * R * E[u_prime(c_{t+1})]
assert np.max(np.abs(euler_residual)) < 1e-6
```

---

## 4. Simulation Consistency

### Common problems
- **Policy function from solver not matching what simulator uses**
- **Random seed management** — results not reproducible
- **Off-by-one in age indexing** between solver (backward) and simulator (forward)
- **Forgetting to apply constraints** during simulation (e.g. borrowing limit)

### What to check
- Simulate a single agent on a deterministic path; verify by hand
- Check that simulated wealth distribution has reasonable moments

---

## 5. Performance Red Flags

### Python-specific
- **Loops over grid points in pure Python** — should be vectorised with numpy
- **Repeated memory allocation** inside iteration loops
- **Redundant recomputation** of things that don't change (transition matrices, grids)
- **Unnecessary copies** (`np.array(x)` when `x` is already an array)

### Algorithmic
- **Solving sub-problems that have closed-form solutions**
- **Iterating to convergence when a fixed number of periods suffices** (finite horizon)
- **Not exploiting sparsity** in transition matrices

---

## 6. Code Organisation Smells

- **God functions** (200+ line functions doing grid setup + solve + simulate)
- **Magic numbers** — unnamed constants in formulas
- **Dead code** from earlier model versions left in
- **Commented-out alternatives** with no explanation of which is correct
- **Copy-paste across files** instead of shared utilities
- **Mutable default arguments** in Python function signatures

---

## 7. Numerical Hygiene

- **Comparing floats with ==** instead of np.isclose
- **Log of zero or negative** without guards
- **Division by values that can be zero**
- **Overflow in exp()** — use logsumexp where applicable
- **Dtype mismatches** (float32 vs float64 silently losing precision)

---

## Review Output Format

When reviewing code, produce a structured report:

```markdown
### File: {filename}

**Issues found:** {count}

| # | Line(s) | Severity | Category | Description | Suggested fix |
|---|---------|----------|----------|-------------|---------------|
| 1 | 42-48   | HIGH     | Grid     | ...         | ...           |
```

Severity levels:
- **CRITICAL** — produces wrong results silently
- **HIGH** — likely bug or major performance issue
- **MEDIUM** — code smell, maintainability risk
- **LOW** — style, minor inefficiency

<!-- TODO: Add any project-specific review patterns once codebase is examined.
     e.g. "In our code, the discount factor is stored as `beta` in params
     but some functions receive `delta` = 1-beta — check for confusion." -->
