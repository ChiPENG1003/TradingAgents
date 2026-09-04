"""Validate frozen snapshots without reading market data or calling an LLM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tradingagents.agents.managers.portfolio_state_manager import (
    FROZEN_ANCHOR_FIELDS,
    FROZEN_MARKET_STATE_FIELDS,
    freeze_feature_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _walk_numbers(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_numbers(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, f"{path}[{index}]")
    elif isinstance(value, float):
        yield path, value


def audit_tree(source: Path) -> dict:
    issues: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    event_impacts: list[float] = []
    files = sorted(source.glob("*/*.json"))
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            frozen = freeze_feature_snapshot(raw)
        except Exception as exc:
            issues.append({"path": str(path), "issue": str(exc)})
            continue
        sample_id = frozen["sample_id"]
        if sample_id in sample_ids:
            issues.append({"path": str(path), "issue": f"duplicate sample_id {sample_id}"})
        sample_ids.add(sample_id)
        if frozen["trade_date"] != frozen["as_of_close_date"]:
            issues.append({"path": str(path), "issue": "trade_date/as_of_close_date mismatch"})
        declared = frozen.get("feature_columns") or {}
        if declared.get("ohlcv_anchors") != list(FROZEN_ANCHOR_FIELDS):
            issues.append({"path": str(path), "issue": "ohlcv feature_columns mismatch"})
        expected_state_columns = list(FROZEN_MARKET_STATE_FIELDS) + ["event_certainty"]
        if declared.get("market_state") != expected_state_columns:
            issues.append({"path": str(path), "issue": "market_state feature_columns mismatch"})
        if set(frozen["ohlcv_anchors"]) != set(FROZEN_ANCHOR_FIELDS):
            issues.append({"path": str(path), "issue": "actual OHLCV fields differ from allowlist"})
        forbidden = {
            "prior_policy_output_do_not_use_as_label", "execution_context",
            "derived_regimes", "event_features", "state_summary", "key_risks",
            "evidence", "invalidation_detail", "rationale_summary",
        }
        present_forbidden = forbidden.intersection(frozen)
        present_forbidden.update(forbidden.intersection(frozen.get("market_state") or {}))
        if present_forbidden:
            issues.append({
                "path": str(path),
                "issue": f"forbidden fields present: {sorted(present_forbidden)}",
            })
        for number_path, value in _walk_numbers(frozen):
            if not math.isfinite(value):
                issues.append({"path": str(path), "issue": f"non-finite {number_path}"})
        event_impacts.append(float(frozen["market_state"]["event_impact_score"]))

    nonzero_event = sum(abs(value) > 1e-12 for value in event_impacts)
    return {
        "schema_version": "frozen_feature_audit_v1",
        "files_checked": len(files),
        "unique_samples": len(sample_ids),
        "issues": issues,
        "event_distribution": {
            "nonzero_direction_count": nonzero_event,
            "nonzero_direction_rate": nonzero_event / len(event_impacts) if event_impacts else 0.0,
            "degenerate_warning": bool(event_impacts) and nonzero_event < max(3, len(event_impacts) // 20),
        },
        "ok": not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen feature snapshots.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "back_test" / "frozen_features")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_tree(args.source)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
