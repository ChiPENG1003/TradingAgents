"""Migrate feature_snapshot_v1 files to immutable quant-only V2 records.

The source tree is never overwritten. By default output is written under
``back_test/frozen_features`` and a hash manifest is produced alongside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingagents.agents.managers.portfolio_state_manager import (
    FROZEN_FEATURE_SCHEMA_VERSION,
    freeze_feature_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def freeze_tree(source: Path, output: Path) -> dict:
    records = []
    seen: dict[str, str] = {}
    for path in sorted(source.glob("*/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        frozen = freeze_feature_snapshot(raw)
        sample_id = frozen["sample_id"]
        digest = frozen["snapshot_hash"]
        if sample_id in seen:
            raise ValueError(f"duplicate canonical sample_id: {sample_id}")
        seen[sample_id] = digest
        ticker_dir = output / frozen["ticker"]
        ticker_dir.mkdir(parents=True, exist_ok=True)
        target = ticker_dir / path.name
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("snapshot_hash") != digest:
                raise FileExistsError(f"refusing to overwrite changed snapshot: {target}")
        else:
            target.write_text(
                json.dumps(frozen, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        records.append({"sample_id": sample_id, "snapshot_hash": digest, "path": str(target)})

    manifest = {
        "schema_version": "frozen_dataset_manifest_v1",
        "feature_schema_version": FROZEN_FEATURE_SCHEMA_VERSION,
        "records": records,
        "record_count": len(records),
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze quant-only backtest feature snapshots.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "back_test" / "features")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "back_test" / "frozen_features")
    args = parser.parse_args()
    manifest = freeze_tree(args.source, args.output)
    print(f"Frozen {manifest['record_count']} snapshots into {args.output}")


if __name__ == "__main__":
    main()
