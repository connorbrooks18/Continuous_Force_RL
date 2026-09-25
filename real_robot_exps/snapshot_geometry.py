"""Structure-snapshot helpers shared by the field session, the pull script and compile.

Snapshots are taken by the running detector (``at-tracking/Detecting.py
--snapshot-dir``). ``request_snapshot`` is the client side of that protocol;
``update_pre_grasp_geometry_with_snapshots`` stores the two pre-grasp snapshots
and derives the spur/stem connection angles from the lengthened one.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

REQUEST_SUFFIX = ".request.json"  # must match at-tracking/snapshot_requests.py


class SnapshotError(RuntimeError):
    """The detector could not produce the requested snapshot."""


def rpy_deg_from_vector(vec: np.ndarray) -> list[float]:
    """Pitch/yaw (deg) that point +x along ``vec``; roll is undefined for a vector."""
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    v = vec / norm
    yaw = float(np.degrees(np.arctan2(v[1], v[0])))
    pitch = float(np.degrees(np.arctan2(-v[2], np.hypot(v[0], v[1]))))
    return [0.0, pitch, yaw]


def update_pre_grasp_geometry_with_snapshots(
    pre_grasp_geometry: dict[str, Any],
    *,
    under_gravity_snapshot: dict[str, Any] | None = None,
    lengthened_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the under-gravity and lengthened snapshots and derive connection angles.

    The lengthened (stretched) snapshot makes segment directions visible, so the
    spur and stem ``connection_rpy_deg`` are computed from it. Snapshots that are
    not given keep whatever the geometry already holds.
    """
    out = json.loads(json.dumps(pre_grasp_geometry))
    under_gravity = under_gravity_snapshot or out.get("under_gravity_snapshot") or {}
    lengthened = lengthened_snapshot or out.get("lengthened_snapshot") or {}
    out["under_gravity_snapshot"] = under_gravity
    out["lengthened_snapshot"] = lengthened

    parts = out.setdefault("parts", {})
    if "primary" in parts:
        parts["primary"]["connection_rpy_deg"] = [0.0, 0.0, 0.0]
    if lengthened and all(key in lengthened for key in ("branch_pos", "spur_pos", "apple_pos")):
        branch = np.asarray(lengthened["branch_pos"], dtype=np.float64)
        spur = np.asarray(lengthened["spur_pos"], dtype=np.float64)
        apple = np.asarray(lengthened["apple_pos"], dtype=np.float64)
        if "spur" in parts:
            parts["spur"]["connection_rpy_deg"] = rpy_deg_from_vector(spur - branch)
            parts["spur"]["connection_source"] = "lengthened_snapshot"
        if "stem" in parts:
            parts["stem"]["connection_rpy_deg"] = rpy_deg_from_vector(apple - spur)
            parts["stem"]["connection_source"] = "lengthened_snapshot"
    if "apple" in parts:
        parts["apple"]["connection_rpy_deg"] = [0.0, 0.0, 0.0]
        parts["apple"]["connection_source"] = "lengthened_snapshot"
    return out


def request_snapshot(
    request_dir: Path | str,
    label: str,
    output: Path | str,
    *,
    frames: int = 5,
    timeout_s: float = 10.0,
    wait_margin_s: float = 10.0,
    detector_alive=None,
) -> dict[str, Any]:
    """Ask the running detector for a median snapshot and wait for the result.

    ``detector_alive`` is an optional callable; when it returns False the wait
    aborts early instead of running into the timeout.
    """
    request_dir = Path(request_dir)
    output = Path(output).resolve()
    request_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    request_path = request_dir / f"{label}-{uuid.uuid4().hex[:8]}{REQUEST_SUFFIX}"
    temporary = request_path.with_name(request_path.name + ".tmp")
    temporary.write_text(
        json.dumps({"label": label, "output": str(output), "frames": int(frames), "timeout_s": float(timeout_s)}),
        encoding="utf-8",
    )
    temporary.replace(request_path)

    deadline = time.monotonic() + float(timeout_s) + float(wait_margin_s)
    while time.monotonic() < deadline:
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if "error" in payload:
                raise SnapshotError(f"{label}: {payload['error']}")
            return payload
        if detector_alive is not None and not detector_alive():
            request_path.unlink(missing_ok=True)
            raise SnapshotError(f"{label}: detector process is not running")
        time.sleep(0.05)
    request_path.unlink(missing_ok=True)
    raise SnapshotError(f"{label}: no answer from the detector within {timeout_s + wait_margin_s:.0f} s")
