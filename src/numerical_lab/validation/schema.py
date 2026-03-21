from __future__ import annotations

from typing import Any, Dict, List, Optional


SEVERITY_PASS = "pass"
SEVERITY_WARNING = "warning"
SEVERITY_SUSPICIOUS = "suspicious"

STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_SUSPICIOUS = "suspicious"


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def make_issue(
    code: str,
    severity: str,
    message: str,
    method: Optional[str] = None,
    expected: Any = None,
    observed: Any = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "method": method,
        "message": message,
        "expected": expected,
        "observed": observed,
        "details": details or {},
    }


def summarize_issues(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {
        SEVERITY_PASS: 0,
        SEVERITY_WARNING: 0,
        SEVERITY_SUSPICIOUS: 0,
    }
    for issue in issues:
        sev = issue.get("severity", SEVERITY_WARNING)
        counts[sev] = counts.get(sev, 0) + 1

    if counts[SEVERITY_SUSPICIOUS] > 0:
        status = STATUS_SUSPICIOUS
    elif counts[SEVERITY_WARNING] > 0:
        status = STATUS_WARNING
    else:
        status = STATUS_PASS

    return {
        "status": status,
        "counts": counts,
        "total_issues": len(issues),
    }


def render_validation_text(validation: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("GRASP Validation Report")
    lines.append("=" * 80)
    lines.append("")

    overview = validation.get("overview", {})
    lines.append(f"Overall status: {overview.get('status', 'unknown')}")
    lines.append(f"Total checks/issues recorded: {overview.get('total_issues', 0)}")
    lines.append(
        "Severity counts: "
        f"pass={overview.get('counts', {}).get('pass', 0)}, "
        f"warning={overview.get('counts', {}).get('warning', 0)}, "
        f"suspicious={overview.get('counts', {}).get('suspicious', 0)}"
    )
    lines.append("")

    for section_name in ("problem_checks", "solver_checks", "consistency_checks"):
        section = validation.get(section_name, {})
        lines.append(section_name.replace("_", " ").title())
        lines.append("-" * 80)

        section_overview = section.get("overview", {})
        lines.append(f"Status: {section_overview.get('status', 'unknown')}")
        lines.append(
            "Counts: "
            f"pass={section_overview.get('counts', {}).get('pass', 0)}, "
            f"warning={section_overview.get('counts', {}).get('warning', 0)}, "
            f"suspicious={section_overview.get('counts', {}).get('suspicious', 0)}"
        )
        lines.append("")

        issues = section.get("issues", [])
        if not issues:
            lines.append("No issues recorded.")
            lines.append("")
            continue

        for i, issue in enumerate(issues, start=1):
            method = issue.get("method")
            prefix = f"[{i}] {issue.get('severity', 'unknown').upper()} | {issue.get('code', 'no_code')}"
            if method:
                prefix += f" | method={method}"
            lines.append(prefix)
            lines.append(f"    {issue.get('message', '')}")

            if "expected" in issue and issue.get("expected") is not None:
                lines.append(f"    expected: {issue.get('expected')}")
            if "observed" in issue and issue.get("observed") is not None:
                lines.append(f"    observed: {issue.get('observed')}")

            details = issue.get("details") or {}
            if details:
                for k, v in details.items():
                    lines.append(f"    {k}: {v}")

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"