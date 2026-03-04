import React from "react";

function normalizeLevel(levelRaw) {
  const v = (levelRaw || "").toString().toLowerCase();
  if (v === "error") return "error";
  if (v === "warn" || v === "warning") return "warn";
  if (v === "info") return "info";
  return null;
}

const ERROR_CODES = new Set([
  "NONFINITE",
  "BAD_BRACKET",
  "DERIV_TOO_SMALL",
  "DERIVATIVE_ZERO",
  "NAN_INF",
  "ERROR",
  "DOMAIN_ERROR",
]);

const WARN_CODES = new Set(["STEP_REJECTED", "STAGNATION", "MAX_ITER"]);

function eventSeverityFromCode(codeRaw) {
  const code = codeRaw ? String(codeRaw).toUpperCase() : "";
  if (!code) return "info";
  if (ERROR_CODES.has(code)) return "error";
  if (WARN_CODES.has(code)) return "warn";
  return "info";
}

function severityStyles(sev) {
  switch (sev) {
    case "error":
      return {
        row: { borderLeft: "4px solid #dc2626", background: "#fef2f2", color: "#7f1d1d" },
        badge: { background: "#fee2e2", color: "#991b1b", border: "1px solid rgba(0,0,0,0.10)" },
      };
    case "warn":
      return {
        row: { borderLeft: "4px solid #f59e0b", background: "#fffbeb", color: "#78350f" },
        badge: { background: "#fef3c7", color: "#92400e", border: "1px solid rgba(0,0,0,0.10)" },
      };
    default:
      return {
        row: { borderLeft: "4px solid #9ca3af", background: "#f9fafb", color: "#111827" },
        badge: { background: "#f3f4f6", color: "#374151", border: "1px solid rgba(0,0,0,0.10)" },
      };
  }
}

export default function EventTimeline({ events, limit }) {
  const list = Array.isArray(events) ? events : [];
  const shown = typeof limit === "number" ? list.slice(0, limit) : list;

  if (shown.length === 0) return <div style={{ color: "#666" }}>No events logged.</div>;

  return (
    <div style={{ border: "1px solid #eee", borderRadius: 10, padding: 10 }}>
      {shown.map((e, idx) => {
        const code = e?.code || e?.data?.code || e?.kind;
        const message = e?.message || e?.data?.message || e?.data?.msg || "";
        const levelRaw = e?.level || e?.data?.level || null;
        const k =
          typeof e?.k === "number" ? e.k : typeof e?.data?.k === "number" ? e.data.k : null;

        const explicit = normalizeLevel(levelRaw);
        const sev = explicit || eventSeverityFromCode(code);
        const styles = severityStyles(sev);

        return (
          <div
            key={idx}
            style={{
              ...styles.row,
              padding: "8px 10px",
              borderRadius: 10,
              margin: "6px 0",
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              fontSize: 12,
              boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
            }}
          >
            <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
              <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, fontWeight: 900, ...styles.badge }}>
                {sev.toUpperCase()}
              </span>
              {k != null ? <span style={{ opacity: 0.75 }}>iter {k}</span> : null}
              <span style={{ fontWeight: 900 }}>{String(code || "—")}</span>
              {e?.kind ? <span style={{ opacity: 0.7 }}>{`(${e.kind})`}</span> : null}
            </div>

            <div style={{ marginTop: 4, opacity: 0.92 }}>
              {message ? message : e?.data ? JSON.stringify(e.data) : ""}
            </div>
          </div>
        );
      })}
    </div>
  );
}