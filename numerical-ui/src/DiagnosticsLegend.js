// src/DiagnosticsLegend.js
import React from "react";

export default function DiagnosticsLegend() {
  return (
    <div style={{ border: "1px solid #eee", borderRadius: 10, padding: 12, marginTop: 10, color: "#333" }}>
      <div style={{ fontWeight: 900, marginBottom: 6 }}>Diagnostics Legend</div>
      <div style={{ fontSize: 13, lineHeight: 1.5 }}>
        <div>
          <strong>Outcome</strong> badges (Converged / Error / Max Iter / Bad Bracket / NaN/Inf / etc.) come from{" "}
          <code>summary.status</code>.
        </div>
        <div>
          <strong>Stop</strong> badges (Exact Root / Tol |f| / Tol x / Tol bracket) come from{" "}
          <code>summary.stop_reason</code> (shown only when converged).
        </div>
        <div>
          <strong>Linear / Quadratic / Superlinear / Sublinear/Irregular</strong> come from{" "}
          <code>summary.convergence_class</code> (shown only when meaningful).
        </div>
        <div>
          <strong>Stable / Unstable</strong> come from <code>summary.stability_label</code>.
        </div>
        <div>
          <strong>Domain Error, Step Rejected, Overflow</strong> come only from explicit diagnostic events (never from explanation text).
        </div>
        <div style={{ marginTop: 6, color: "#555" }}>
          <strong>Teaching hints</strong> (when enabled) are deterministic 1-line summaries derived from <code>summary</code> + <code>events</code>.
        </div>
      </div>
    </div>
  );
}