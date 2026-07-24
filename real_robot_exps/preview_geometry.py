"""Print geometry metadata and the first two data rows from a Parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def _read_metadata(path: Path) -> dict:
    payload = (pq.read_schema(path).metadata or {}).get(b"dataset_metadata")
    return json.loads(payload.decode("utf-8")) if payload else {}


def _first_data_rows(path: Path, limit: int = 2) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    data_rows = [row for row in rows if str(row.get("row_kind", "data")) != "metadata"]
    return data_rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Parquet file to inspect")
    args = parser.parse_args()

    metadata = _read_metadata(args.parquet)
    print("pre_grasp_geometry:")
    print(json.dumps(metadata.get("pre_grasp_geometry", {}), indent=2, sort_keys=True))
    print("\npost_grasp_geometry:")
    print(json.dumps(metadata.get("post_grasp_geometry", {}), indent=2, sort_keys=True))
    print("\nfirst 2 data rows:")
    for idx, row in enumerate(_first_data_rows(args.parquet, limit=2)):
        print(f"\nrow {idx}:")
        print(json.dumps(row, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
