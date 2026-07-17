"""Print Parquet metadata, schema, and the first N rows.

Usage:
    python -m real_robot_exps.dump_parquet_preview file.parquet
    python -m real_robot_exps.dump_parquet_preview file.parquet --rows 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def _decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    decoded: dict[str, Any] = {}
    for key, value in metadata.items():
        key_str = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
            try:
                decoded[key_str] = json.loads(text)
            except json.JSONDecodeError:
                decoded[key_str] = text
        else:
            decoded[key_str] = value
    return decoded


def dump_preview(path: Path, rows: int = 15) -> None:
    path = Path(path)
    schema = pq.read_schema(path)
    table = pq.read_table(path, columns=schema.names)
    preview = table.slice(0, rows).to_pandas()

    print(f"File: {path.resolve()}")
    print(f"Rows in file: {table.num_rows}")
    print(f"Columns: {table.num_columns}")
    print("\nSchema:")
    print(schema)

    print("\nParquet footer metadata:")
    metadata = _decode_metadata(schema.metadata)
    if metadata:
        for key, value in metadata.items():
            print(f"{key}:")
            if isinstance(value, (dict, list)):
                print(json.dumps(value, indent=2, sort_keys=True, default=str))
            else:
                print(value)
    else:
        print("(none)")

    print(f"\nFirst {min(rows, len(preview))} rows:")
    if preview.empty:
        print("(no rows)")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 60):
            print(preview.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Parquet file to inspect")
    parser.add_argument("--rows", type=int, default=15, help="Number of rows to print")
    args = parser.parse_args()
    dump_preview(args.parquet, rows=args.rows)


if __name__ == "__main__":
    main()
