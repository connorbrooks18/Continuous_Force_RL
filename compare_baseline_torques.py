"""Compare torque statistics between new and archived unified Parquets.

The script compares ``sXX-dYY.parquet`` with its corresponding
``sXX-dYY_old.parquet`` in ``s02`` through ``s10``. These unified files contain
the baseline-subtracted ``ft_wrist`` values. Torque is read from the final
three values of that field and reported as x, y, z, and vector magnitude.

Examples:

    python compare_baseline_torques.py
    python compare_baseline_torques.py --csv subtracted_torque_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


STRUCTURE_DIRS = tuple(f"s{index:02d}" for index in range(2, 11))
TORQUE_NAMES = ("x", "y", "z", "magnitude")


def _torques(path: Path) -> np.ndarray:
    rows = pq.read_table(path, columns=["ft_wrist"]).to_pylist()
    values = []
    for row in rows:
        wrench = row.get("ft_wrist")
        if wrench is None:
            continue
        array = np.asarray(wrench, dtype=np.float64).reshape(-1)
        if array.size < 6:
            raise ValueError(f"{path} contains an ft_wrist value with fewer than 6 elements")
        values.append(array[3:6])
    if not values:
        raise ValueError(f"{path} contains no usable ft_wrist rows")
    torque = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(torque).all(axis=1)
    torque = torque[valid]
    if not len(torque):
        raise ValueError(f"{path} contains no finite torque rows")
    return np.column_stack((torque, np.linalg.norm(torque, axis=1)))


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(values, axis=0), np.median(values, axis=0)


def _row(label: str, new: np.ndarray, old: np.ndarray) -> dict[str, object]:
    new_mean, new_median = _stats(new)
    old_mean, old_median = _stats(old)
    result: dict[str, object] = {"file": label, "new_samples": len(new), "old_samples": len(old)}
    for index, name in enumerate(TORQUE_NAMES):
        result[f"new_mean_{name}"] = new_mean[index]
        result[f"old_mean_{name}"] = old_mean[index]
        result[f"delta_mean_{name}"] = new_mean[index] - old_mean[index]
        result[f"new_median_{name}"] = new_median[index]
        result[f"old_median_{name}"] = old_median[index]
        result[f"delta_median_{name}"] = new_median[index] - old_median[index]
    return result


def collect_comparisons(root: Path) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    missing: list[str] = []
    for directory_name in STRUCTURE_DIRS:
        directory = root / directory_name
        for old_path in sorted(directory.glob("*-d*_old.parquet")):
            new_name = old_path.name.removesuffix("_old.parquet") + ".parquet"
            new_path = directory / new_name
            if not new_path.is_file():
                missing.append(str(new_path))
                continue
            comparisons.append(_row(f"{directory_name}/{new_name}", _torques(new_path), _torques(old_path)))
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise ValueError(f"Missing new baseline for archived file(s):\n{formatted}")
    if not comparisons:
        raise ValueError("No unified *_old.parquet files found")
    return comparisons


def _print_table(rows: list[dict[str, object]]) -> None:
    print("file | samples new/old | mean torque x y z magnitude (new / old / delta) | median torque x y z magnitude (new / old / delta)")
    for row in rows:
        mean = " ".join(
            f"{row[f'new_mean_{name}']:.6g}/{row[f'old_mean_{name}']:.6g}/{row[f'delta_mean_{name}']:.6g}"
            for name in TORQUE_NAMES
        )
        median = " ".join(
            f"{row[f'new_median_{name}']:.6g}/{row[f'old_median_{name}']:.6g}/{row[f'delta_median_{name}']:.6g}"
            for name in TORQUE_NAMES
        )
        print(f"{row['file']} | {row['new_samples']}/{row['old_samples']} | {mean} | {median}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--csv", type=Path, help="Also write the detailed comparison as CSV")
    args = parser.parse_args()
    root = args.root.resolve()

    rows = collect_comparisons(root)
    pooled_new = []
    pooled_old = []
    for directory_name in STRUCTURE_DIRS:
        directory = root / directory_name
        for old_path in sorted(directory.glob("*-d*_old.parquet")):
            new_path = directory / (old_path.name.removesuffix("_old.parquet") + ".parquet")
            if new_path.is_file():
                pooled_new.append(_torques(new_path))
                pooled_old.append(_torques(old_path))
    rows.append(_row("TOTAL (pooled samples)", np.concatenate(pooled_new), np.concatenate(pooled_old)))

    _print_table(rows)
    if args.csv:
        _write_csv(args.csv if args.csv.is_absolute() else root / args.csv, rows)


if __name__ == "__main__":
    main()
