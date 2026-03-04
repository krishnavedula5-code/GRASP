// src/diagnosticsBadges.js
// Deterministic diagnostics engine.
// Inputs: summary + trace.events only.

function toStr(x) {
  return x == null ? "" : String(x);
}
function normUpper(x) {
  return toStr(x).trim().toUpperCase();
}
function normLower(x) {
  return toStr(x).trim().toLowerCase();
}
function titleCase(x) {
  const s = toStr(x).trim();
  if (!s) return "";
  return s.charAt(0).toUpperCase() + s.slice(1);
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

function anyCode(events, codes) {
  return Array.isArray(codes) && codes.some((c) => hasEventCode(events, c));
}
function anyKind(events, kinds) {
  return Array.isArray(kinds) && kinds.some((k) => hasEventKind(events, k));
}

// Strong, deterministic key (case-insensitive + trims) to prevent duplicates
function uniqStable(arr) {
  const seen = new Set();
  const out = [];
  for (const x of arr) {
    const s = String(x ?? "").trim();
    if (!s) continue;
    const key = s.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

// Only allow explicit diagnostic signals to generate badges.
const DIAG = {
  NONFINITE: ["NONFINITE", "NAN_INF", "NAN", "INF"],
  DOMAIN: ["DOMAIN_ERROR"],
  OVERFLOW: ["OVERFLOW", "UNDERFLOW"],
  BAD_BRACKET: ["BAD_BRACKET"],
  STEP_REJECTED: ["STEP_REJECTED"],
  LINESEARCH_FAIL: ["LINESEARCH_FAIL"],
  SINGULAR: ["SINGULAR_JACOBIAN", "SINGULAR_MATRIX", "SINGULAR"],
  NO_PROGRESS: ["NO_PROGRESS", "STAGNATION"],
};

const DIAG_KINDS_FALLBACK = {
  NONFINITE: ["nonfinite", "nan", "inf"],
  DOMAIN: ["domain_error"],
  OVERFLOW: ["overflow", "underflow"],
  BAD_BRACKET: ["bad_bracket", "invalid_bracket"],
  STEP_REJECTED: ["step_rejected"],
  LINESEARCH_FAIL: ["linesearch_fail"],
  SINGULAR: ["singular_jacobian", "singular_matrix", "singular"],
  NO_PROGRESS: ["no_progress", "stagnation"],
};

function outcomeFromStatus(statusLower) {
  const status = normLower(statusLower);
  if (status === "converged") return "Converged";
  if (status === "error") return "Error";
  if (status === "max_iter") return "Max Iter";
  if (status === "bad_bracket") return "Bad Bracket";
  if (status === "nan_or_inf") return "NaN/Inf";
  if (status === "derivative_zero") return "Derivative Zero";
  if (status === "stagnation") return "Stagnation";
  if (status === "diverged") return "Diverged";
  if (status === "stopped") return "Stopped";
  return "Unknown";
}

export function computeBadges(summary, events) {
  const s = summary || {};
  const ev = Array.isArray(events) ? events : [];

  const method = normLower(s.method);
  const status = normLower(s.status);
  const stopReason = normUpper(s.stop_reason);
  const convClass = normLower(s.convergence_class);
  const stability = normLower(s.stability_label);
  const iters = Number.isFinite(Number(s.iterations)) ? Number(s.iterations) : null;

  const out = [];

  // ---------- 1) Outcome: EXACTLY ONE ----------
  const outcome = outcomeFromStatus(status);
  out.push(outcome);

  const isConverged = outcome === "Converged";

  // ---------- 2) Stop reason: at most one, only when converged ----------
  if (isConverged) {
    if (stopReason === "EXACT_ROOT") out.push("Exact Root");
    else if (stopReason === "TOL_F") out.push("Tol |f|");
    else if (stopReason === "TOL_X") out.push("Tol x");
    else if (stopReason === "TOL_BRACKET") out.push("Tol bracket");
  }

  // ---------- 3) Convergence class: only meaningful + converged ----------
  // Show when we have enough iterations to estimate order (keep this low to be useful)
  const MIN_ITERS_FOR_ORDER = 3;

  const convMeaningful =
    isConverged &&
    convClass &&
    convClass !== "unknown" &&
    convClass !== "insufficient_data" &&
    stopReason !== "EXACT_ROOT" &&
    (iters == null || iters >= MIN_ITERS_FOR_ORDER);

 if (convMeaningful) {
  // Normalize backend vocabulary → clean UI labels
  if (convClass === "linear") out.push("Linear");
  else if (convClass === "superlinear") out.push("Superlinear");
  else if (convClass === "quadratic") out.push("Quadratic");
  else if (convClass === "quadratic_or_better") out.push("Quadratic");
  else if (convClass === "sublinear_or_irregular") out.push("Sublinear/Irregular");
  else out.push(titleCase(convClass).replaceAll("_", " "));
}

  // ---------- 4) Stability: only meaningful ----------
  const stabMeaningful =
    stability &&
    stability !== "unknown" &&
    stability !== "insufficient_data" &&
    stopReason !== "EXACT_ROOT";

  if (stabMeaningful) {
    if (stability === "stable") out.push("Stable");
    else if (stability === "unstable") out.push("Unstable");
    else out.push(titleCase(stability));
  }

  // ---------- 5) Event-driven badges (strict whitelist, capped) ----------
  // Avoid re-stating the same thing outcome already says.
  const wantEventBadges = [];
  const hasNonfinite = anyCode(ev, DIAG.NONFINITE) || anyKind(ev, DIAG_KINDS_FALLBACK.NONFINITE);
  const hasDomain = anyCode(ev, DIAG.DOMAIN) || anyKind(ev, DIAG_KINDS_FALLBACK.DOMAIN);
  const hasOverflow = anyCode(ev, DIAG.OVERFLOW) || anyKind(ev, DIAG_KINDS_FALLBACK.OVERFLOW);
  const hasBadBracket = anyCode(ev, DIAG.BAD_BRACKET) || anyKind(ev, DIAG_KINDS_FALLBACK.BAD_BRACKET);
  const hasLineSearchFail = anyCode(ev, DIAG.LINESEARCH_FAIL) || anyKind(ev, DIAG_KINDS_FALLBACK.LINESEARCH_FAIL);
  const hasSingular = anyCode(ev, DIAG.SINGULAR) || anyKind(ev, DIAG_KINDS_FALLBACK.SINGULAR);
  const hasNoProgress = anyCode(ev, DIAG.NO_PROGRESS) || anyKind(ev, DIAG_KINDS_FALLBACK.NO_PROGRESS);

  // Step rejected must be explicit. Do not infer it from other codes.
  const stepRejected = hasEventCode(ev, "STEP_REJECTED") || hasEventKind(ev, "step_rejected");

  // Priority list (highest signal first)
  if (hasDomain) wantEventBadges.push("Domain Error");
  if (hasNonfinite && outcome !== "NaN/Inf") wantEventBadges.push("NaN/Inf"); // don't duplicate outcome
  if (hasOverflow) wantEventBadges.push("Overflow");

  // If solver didn't already label bracket as bad, event can add it.
  if (hasBadBracket && outcome !== "Bad Bracket") {
    wantEventBadges.push(isConverged ? "No sign change" : "Bad Bracket");
  }

  if (stepRejected && method === "hybrid") wantEventBadges.push("Step Rejected");
  if (hasLineSearchFail) wantEventBadges.push("Line Search Fail");
  if (hasSingular) wantEventBadges.push("Singular");
  if (hasNoProgress && outcome !== "Stagnation") wantEventBadges.push("No Progress");

  // Cap event badges to avoid spam (tune if you want 3)
  const MAX_EVENT_BADGES = 2;
  out.push(...wantEventBadges.slice(0, MAX_EVENT_BADGES));

  // ---------- Deterministic ordering ----------
  const order = [
    // outcome first
    "Error",
    "NaN/Inf",
    "Bad Bracket",
    "Derivative Zero",
    "Stagnation",
    "Max Iter",
    "Diverged",
    "Stopped",
    "Converged",
    "Unknown",

    // stop reason
    "Exact Root",
    "Tol |f|",
    "Tol x",
    "Tol bracket",

    // convergence
    "Quadratic",
    "Superlinear",
    "Linear",
    "Sublinear/Irregular",

    // stability
    "Stable",
    "Unstable",

    // events
    "Domain Error",
    "Overflow",
    "No sign change",
    "Line Search Fail",
    "Singular",
    "No Progress",
    "Step Rejected",
  ];

  const unique = uniqStable(out);

  unique.sort((a, b) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });

  return unique;
}