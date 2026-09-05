"""Bounded handoffs; full reports and debate transcripts stay in graph state."""
import json


def bounded_text(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    notice = "\n[excerpt omitted; consult full report for details]\n"
    room = max(0, limit - len(notice))
    return text[:room // 2] + notice + text[-(room - room // 2):]


def evidence_brief(state: dict) -> str:
    """Share factor evidence already extracted by the existing Conflict Detector."""
    report = state.get("conflict_report") or {}
    signals = report.get("signals", [])
    if signals:
        lines = [f"{s['factor']}={s['direction']}; source={s['source_report']}: {bounded_text(s.get('excerpt', ''), 300)}" for s in signals[:6]]
    else:
        lines = [f"{key}: {bounded_text(state.get(key) or '(unavailable)', 650)}" for key in ("market_report", "news_report", "sentiment_report", "fundamentals_report")]
    if state.get("decision_anchors"):
        lines.append("Computed target market anchors: " + json.dumps(state["decision_anchors"], default=str))
    return "\n".join(lines)


def risk_handoff(risk_state: dict) -> str:
    lines = []
    for role in ("aggressive", "conservative", "neutral"):
        response = risk_state.get(f"current_{role}_response")
        lines.append(f"{role}: {bounded_text(response or '(no assessment)', 1400)}")
    return "\n".join(lines)
