// src/diagnosticsHints.js
// Deterministic 1-line teaching hints derived from summary + events only.
// NO explanation parsing.

function normUpper(x) {
  return String(x ?? "").trim().toUpperCase();
}
function normLower(x) {
  return String(x ?? "").trim().toLowerCase();
}

function hasEventCode(events, code) {
  const want = normUpper(code);
  if (!Array.isArray(events) || !want) return false;
  return events.some((e) => normUpper(e?.code || e?.data?.code) === want);
}
function hasEventKind(events, kind) {
  const want = normLower(kind);
  if (!Array.isArray(events) || !want) return false;
  return events.some((e) => normLower(e?.kind || e?.data?.kind) === want);
}

function countEventCode(events, code) {
  const want = normUpper(code);
  if (!Array.isArray(events) || !want) return 0;
  return events.reduce((acc, e) => acc + (normUpper(e?.code || e?.data?.code) === want ? 1 : 0), 0);
}

// Prefer message-level facts in events when they exist
function findFirstEventWithData(events, codesUpper) {
  const want = new Set((codesUpper || []).map(normUpper));
  if (!Array.isArray(events)) return null;
  for (const e of events) {
    const c = normUpper(e?.code || e?.data?.code);
    if (want.has(c)) return e;
  }
  return null;
}

export function computeHint(methodKey, summary, events) {
  const m = normLower(methodKey || summary?.method);
  const s = summary || {};
  const ev = Array.isArray(events) ? events : [];

  const status = normLower(s.status);
  const stopReason = normUpper(s.stop_reason);
  const convClass = normLower(s.convergence_class);
  const iters = Number.isFinite(Number(s.iterations)) ? Number(s.iterations) : null;

  // --- High-signal failure hints (top priority) ---
  const hasDomain =
    hasEventCode(ev, "DOMAIN_ERROR") ||
    hasEventCode(ev, "NONFINITE") ||
    hasEventCode(ev, "NAN_INF") ||
    hasEventKind(ev, "domain_error") ||
    hasEventKind(ev, "nonfinite");

  if (status === "nan_or_inf" || hasDomain) {
    return "Non-finite evaluation (domain violation or overflow) — check bracket/inputs.";
  }

  const hasBadBracket = hasEventCode(ev, "BAD_BRACKET") || hasEventKind(ev, "bad_bracket") || hasEventKind(ev, "invalid_bracket");
  if (status === "bad_bracket" || hasBadBracket) {
    return "No sign change on bracket — bisection/hybrid require opposite-sign endpoints.";
  }

  const hasDerivZero =
    status === "derivative_zero" ||
    hasEventCode(ev, "DERIVATIVE_ZERO") ||
    hasEventCode(ev, "DERIV_TOO_SMALL") ||
    hasEventKind(ev, "derivative_zero");

  if (hasDerivZero) {
    return "Derivative too small/zero — Newton step becomes unreliable near flat regions.";
  }

  const hasStagnation = status === "stagnation" || hasEventCode(ev, "STAGNATION") || hasEventKind(ev, "stagnation");
  if (hasStagnation) {
    return "Stagnation detected — progress stalled (try different initial guesses or method).";
  }

  if (status === "max_iter" || stopReason === "MAX_ITER" || hasEventCode(ev, "MAX_ITER")) {
    return "Hit max iterations — consider a tighter bracket, better initial guess, or looser tolerance.";
  }

  if (status === "error" || stopReason === "ERROR") {
    return "Solver reported an error — check function/derivative validity and numeric stability.";
  }

  // --- Converged hints (method-specific) ---
  if (status === "converged") {
    if (stopReason === "EXACT_ROOT") return "Exact root hit (residual exactly zero).";
    if (stopReason === "TOL_F") return "Stopped when |f(x)| met tolerance.";
    if (stopReason === "TOL_X") return "Stopped when step size met tolerance.";
    if (stopReason === "TOL_BRACKET") return "Stopped when bracket width met tolerance.";

    // Hybrid: focus on step acceptance/rejection
    if (m === "hybrid") {
      const rejCount = countEventCode(ev, "STEP_REJECTED") + countEventCode(ev, "NEWTON_REJECT");
      if (hasEventCode(ev, "STEP_REJECTED") || hasEventKind(ev, "step_rejected") || rejCount > 0) {
        const e = findFirstEventWithData(ev, ["STEP_REJECTED"]);
        const reason = e?.data?.reason || e?.data?.reject_reason || null;
        if (reason === "step_outside_interval") return "Rejected Newton steps to stay inside the bracket (safety).";
        return "Rejected some Newton steps — hybrid fell back to bracketing for robustness.";
      }
      return "Hybrid combines bracketing safety with Newton speed when acceptable.";
    }

    // Bisection: always linear + guaranteed when bracket valid
    if (m === "bisection") {
      return "Guaranteed convergence on valid bracket, but linear (slower than Newton/Secant).";
    }

    // Newton: fast near root, needs derivative + good guess
    if (m === "newton") {
      if (convClass === "linear") return "Likely multiple/flat root — Newton becomes linear near the solution.";
      return "Fast near the root (often quadratic), but needs a good initial guess and nonzero derivative.";
    }

    // Secant: derivative-free, can be unstable if denominator collapses
    if (m === "secant") {
      const denomSmall =
        hasEventCode(ev, "DENOM_TOO_SMALL") ||
        hasEventKind(ev, "denom_too_small") ||
        hasEventCode(ev, "STAGNATION");
      if (denomSmall) return "Secant can stall if f(x1)≈f(x0); try different guesses.";
      return "Fast without derivative; may be unstable if guesses cause near-zero denominator.";
    }

    // Generic converged fallback
    if (iters != null && iters < 3) return "Converged quickly — limited data to assess convergence order.";
    return "Converged successfully.";
  }

  // --- Generic fallback ---
  return null;
}