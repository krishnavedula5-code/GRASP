# Numerical Lab Report

_Generated: 2026-02-23T07:44:50_

## Problem

`f(x) = x**3 - x - 2`

## Method Summary

| Method | Status | Iters | Root | Last |f(x)| | Order | Stability |
|---|---|---:|---:|---:|---:|---|
| bisection | converged | 33 | 1.5213797068 | 3.278e-11 | 1.000 | possible_oscillation |
| newton | converged | 3 | 1.5213797068 | 4.530e-14 | 1.998 | stable |
| secant | converged | 8 | 1.5213797068 | 1.843e-14 | 1.629 | possible_oscillation |
| hybrid | converged | 30 | 1.52137970682 | 6.902e-11 | 0.000 | possible_oscillation |

## Explanations

### bisection

BISECTION — status: converged. Root ≈ 1.5213797068. Iterations: 33. Final |f(x)| = 3.278e-11. Residual improved by ~3.8e+09× overall. Step sizes shrank significantly (approaching the root). Observed convergence: linear. Estimated order p ≈ 1.000. Bisection is guaranteed with a valid bracket; it converges linearly by halving the interval. Stability flag: possible oscillation. Solver reports convergence. Observed order ≈ 1.000 (linear convergence). Frequent direction changes detected; possible oscillation.

### newton

NEWTON — status: converged. Root ≈ 1.5213797068. Iterations: 3. Final |f(x)| = 4.530e-14. Residual improved by ~2.8e+12× overall. Observed convergence: quadratic or better. Estimated order p ≈ 1.998. Newton is fast near the root but needs a good initial guess and a nonzero derivative. Solver reports convergence. Observed order ≈ 1.998 (quadratic or better convergence). No strong instability patterns detected.

### secant

SECANT — status: converged. Root ≈ 1.5213797068. Iterations: 8. Final |f(x)| = 1.843e-14. Residual improved by ~1.1e+14× overall. Residual decreased consistently (stable progress). Step sizes shrank significantly (approaching the root). Observed convergence: superlinear. Estimated order p ≈ 1.629. Secant is derivative-free and often superlinear, but can be unstable for poor starting points. Stability flag: possible oscillation. Solver reports convergence. Observed order ≈ 1.629 (superlinear convergence). Frequent direction changes detected; possible oscillation.

### hybrid

HYBRID — status: converged. Root ≈ 1.52137970682. Iterations: 30. Final |f(x)| = 6.902e-11. Residual improved by ~3.1e+07× overall. Step sizes shrank significantly (approaching the root). Estimated order p ≈ 0.000. Hybrid keeps a bracket for safety and uses Newton steps when they appear reliable. Hybrid behavior: Newton accepted 14/30 attempts; bisection fallback used 16 times. Top Newton rejection reasons: step_outside_bracket:16. Action: Newton step left the safe bracket — hybrid/bisection is appropriate; also try closer initial guess. Stability flag: possible oscillation. Solver reports convergence. Observed order ≈ 0.000 (unusual/unstable estimate). Frequent direction changes detected; possible oscillation.
