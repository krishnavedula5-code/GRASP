import React from "react";

/**
 * BadgePills
 * - Renders badges with clear spacing (no "word concatenation")
 * - De-duplicates badges case-insensitively (prevents Converged + CONVERGED)
 * - Does NOT inject outcome/status; it renders ONLY what it receives via props.
 * - Shows overflow as +N with tooltip
 */

function normKey(x) {
  return String(x ?? "").trim().toLowerCase();
}

export default function BadgePills({ badges, max = 6 }) {
  const list = Array.isArray(badges) ? badges : [];

  // De-dupe (case-insensitive) while preserving order
  const seen = new Set();
  const unique = [];
  for (const b of list) {
    const s = String(b ?? "").trim();
    if (!s) continue;

    const key = normKey(s);
    if (seen.has(key)) continue;

    seen.add(key);
    unique.push(s);
  }

  const shown = unique.slice(0, Math.max(0, max));
  const remaining = unique.length - shown.length;

  if (shown.length === 0) return null;

  const pillStyle = {
    display: "inline-block",
    fontSize: 11,
    fontWeight: 800,
    padding: "3px 8px",
    borderRadius: 999,
    border: "1px solid rgba(0,0,0,0.18)",
    background: "rgba(0,0,0,0.04)",
    lineHeight: 1.2,
  };

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
      {shown.map((b, idx) => (
        <span key={`${normKey(b)}-${idx}`} style={pillStyle}>
          {b}
        </span>
      ))}

      {remaining > 0 ? (
        <span style={pillStyle} title={unique.slice(shown.length).join(", ")}>
          +{remaining}
        </span>
      ) : null}
    </div>
  );
}