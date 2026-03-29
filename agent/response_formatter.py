from __future__ import annotations

from typing import Iterable

from backend.schemas import AgentResponse, Evidence, MissingData


def format_agent_response(response: AgentResponse) -> str:
    """
    Return a human-friendly multiline rendering of an AgentResponse.

    This formatter is intended for demos, terminal display, and quick operator
    review. It does not change the underlying structured payload.
    """
    lines: list[str] = []
    lines.append("=== Diagnostic V2 ===")
    lines.append(f"Request ID : {response.request_id}")
    lines.append(f"Vehicle ID : {response.vehicle_id}")
    lines.append(f"Intent     : {response.intent.name} (confidence intent={response.intent.confidence:.2f})")
    if response.intent.parameters.goal:
        lines.append(f"Goal       : {response.intent.parameters.goal} [{response.intent.parameters.scope or 'specific'}]")
    if response.intent.parameters.resolution_policy:
        lines.append(f"Policy     : {response.intent.parameters.resolution_policy}")
    lines.append(f"Confidence : {response.confidence:.2f}")
    lines.append("")
    lines.append("Diagnostic")
    lines.append(response.diagnosis)

    if response.evidence:
        lines.append("")
        lines.append("Evidence")
        lines.extend(_format_evidence(response.evidence))

    if response.signals_used:
        lines.append("")
        lines.append("Signals Used")
        lines.append(", ".join(response.signals_used))

    if response.actions_taken:
        lines.append("")
        lines.append("Actions Taken")
        lines.append(" -> ".join(response.actions_taken))

    lines.append("")
    lines.append("Missing Data")
    if response.missing_data:
        lines.extend(_format_missing_data(response.missing_data))
    else:
        lines.append("None")

    if response.recommendations:
        lines.append("")
        lines.append("Recommendations")
        for item in response.recommendations:
            lines.append(f"- {item}")

    return "\n".join(lines)


def _format_evidence(evidence: Iterable[Evidence]) -> list[str]:
    rows: list[str] = []
    for item in evidence:
        unit = f" {item.unit}" if item.unit else ""
        rows.append(
            f"- {item.label}: {item.value}{unit} "
            f"[source={item.source}, observed_ts={item.observed_ts.isoformat()}]"
        )
    return rows


def _format_missing_data(items: Iterable[MissingData]) -> list[str]:
    rows: list[str] = []
    for item in items:
        impact = f", impact={item.impact}" if item.impact else ""
        rows.append(f"- {item.key}: reason={item.reason}{impact}")
    return rows
