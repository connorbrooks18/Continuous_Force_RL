"""Check that everything an apple needs was saved, and that it is usable.

    python -m real_robot_exps.field_session --session S --verify A002   (or all)

Read-only. Each section reports PASS / WARN / FAIL with details, and the report
ends with what to redo. FAIL means compile (or the data's purpose) is blocked;
WARN means the data is usable but something should be looked at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}
ROBOT_TOP_LEVEL_KEYS = (
    "episode_id", "rest_reference_timestamp", "theta_rad", "phi_rad", "robot_start_pose_4x4",
    "robot_start_joint_pos", "pre_grasp_geometry", "post_grasp_geometry", "dump",
)
TRACKERS = ("Branch", "Spur", "Apple")


@dataclass
class Section:
    name: str
    status: str = "PASS"
    lines: list[str] = field(default_factory=list)

    def note(self, status: str, text: str) -> None:
        self.lines.append(f"{status:4s} {text}")
        if RANK[status] > RANK[self.status]:
            self.status = status


@dataclass
class Report:
    apple_id: str
    sections: list[Section] = field(default_factory=list)
    redo: list[str] = field(default_factory=list)

    def section(self, name: str) -> Section:
        s = Section(name)
        self.sections.append(s)
        return s

    @property
    def status(self) -> str:
        return max((s.status for s in self.sections), key=RANK.__getitem__, default="PASS")

    def text(self) -> str:
        out = [f"===== {self.apple_id}: {self.status}"]
        for s in self.sections:
            out.append(f"[{s.status}] {s.name}")
            out.extend(f"    {line}" for line in s.lines)
        if self.redo:
            out.append("To fix:")
            out.extend(f"    - {item}" for item in dict.fromkeys(self.redo))
        return "\n".join(out)


def _metadata(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    raw = (pq.read_schema(path).metadata or {}).get(b"dataset_metadata")
    return json.loads(raw) if raw else {}


def _data_rows(path: Path):
    import pyarrow.parquet as pq

    frame = pq.read_table(path).to_pandas()
    if "row_kind" in frame:
        frame = frame[frame["row_kind"] != "metadata"]
    return frame


def verify_apple(apple_dir: Path, session_data: dict[str, Any], session_name: str = "S") -> Report:
    apple_dir = Path(apple_dir)
    apple = json.loads((apple_dir / "apple.json").read_text(encoding="utf-8"))
    apple_id = apple_dir.name
    report = Report(apple_id)
    redo = lambda *steps: report.redo.append(
        f"python -m real_robot_exps.field_session --session {session_name} --apple {apple_id} "
        + " ".join(f"--redo {step}" for step in steps)
    )
    directions = [d["index"] for d in session_data.get("directions", [])]
    stops = int(session_data.get("stops", 4))

    # --- steps ------------------------------------------------------------------------
    sec = report.section("steps")
    for name, info in apple.get("steps", {}).items():
        status = info.get("status")
        if name == "tool_and_tags":
            continue  # renamed to 'tags'; old sessions keep the stale entry
        if status == "done":
            continue
        sec.note("WARN" if status == "skipped" else "FAIL", f"{name}: {status} {info.get('error', '')}".strip())

    # --- calibration -------------------------------------------------------------------
    sec = report.section("calibration")
    calib = apple_dir / "calib"
    calibration = apple.get("calibration", {})
    verdict = calibration.get("verdict")
    for pattern in ("*.calib", "*.samples", "report.txt", "camera_link_to_optical.json"):
        if verdict == "SKIPPED" and pattern != "report.txt":
            continue
        if not list(calib.glob(pattern)) and verdict != "SKIPPED":
            sec.note("FAIL", f"missing calib/{pattern}")
    camera_to_base = None
    try:
        from real_robot_exps.frame_transforms import load_camera_to_base

        camera_to_base = load_camera_to_base(calib / "camera_to_base.json")
        sec.note("PASS", f"camera_to_base.json valid; camera at {np.round(camera_to_base[:3, 3], 3).tolist()} m")
    except Exception as exc:
        sec.note("FAIL", f"camera_to_base.json unusable: {exc}")
        redo("calibrate")
    if verdict == "GOOD":
        sec.note("PASS", "verdict GOOD")
    elif verdict in ("MARGINAL", "BAD", "UNKNOWN"):
        sec.note("WARN", f"verdict {verdict}" + (" (accepted anyway)" if calibration.get("accepted_despite_verdict") else ""))
    elif verdict == "SKIPPED":
        sec.note("WARN", "calibration skipped: static camera matrix used")
    else:
        sec.note("FAIL", "no calibration verdict recorded")

    # --- snapshots ---------------------------------------------------------------------
    sec = report.section("snapshots")
    for label in ("under_gravity", "lengthened"):
        path = apple_dir / "snapshots" / f"{label}.json"
        if not path.exists():
            sec.note("FAIL", f"{label}.json missing")
            redo("snapshots")
            continue
        snapshot = json.loads(path.read_text())
        if "error" in snapshot or not all(f"{k}_pos" in snapshot for k in ("apple", "branch", "spur")):
            sec.note("FAIL", f"{label}: {snapshot.get('error', 'incomplete')}")
            redo("snapshots")
        else:
            png = "with image" if path.with_suffix(".png").exists() else "no image"
            sec.note("PASS", f"{label}: {snapshot.get('camera_frame_count')} frames, all 3 tags, {png}")
    post_errors = []
    for index in directions:
        path = apple_dir / "snapshots" / f"post_grasp_d{index:02d}.json"
        if path.exists() and "error" in json.loads(path.read_text()):
            post_errors.append(f"d{index:02d}")
    if post_errors:
        sec.note("WARN", f"post-grasp snapshot failed for {', '.join(post_errors)} (a tag was hidden with the apple held)")

    # --- tracking + video --------------------------------------------------------------
    sec = report.section("tracking + video")
    trackings = sorted((apple_dir / "tracking").glob("tracking_*.parquet"))
    tracking_frames = []
    if not trackings:
        sec.note("FAIL", "no tracking file")
    for tracking in trackings:
        meta = _metadata(tracking)
        frame = _data_rows(tracking)
        stamps = np.sort(frame["timestamp"].unique())
        tracking_frames.append((tracking, frame, stamps))
        span = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        sec.note("PASS", f"{tracking.name}: {len(stamps)} frames, {span:.0f} s, "
                         f"{len(stamps) / span if span else 0:.1f} fps, final={not meta.get('partial', False)}")
        used = meta.get("camera_to_base_4x4_used")
        if camera_to_base is not None and used is not None and not np.allclose(np.asarray(used), camera_to_base, atol=1e-9):
            sec.note("FAIL", f"{tracking.name} used a different camera matrix than calib/camera_to_base.json")
        if meta.get("video_recording_enabled"):
            video = Path(meta.get("video_recording_path") or tracking.with_suffix(".mp4"))
            times = Path(str(video) + ".timestamps.csv")
            if not video.exists():
                sec.note("FAIL", f"video {video.name} missing")
            elif not times.exists():
                sec.note("WARN", f"{video.name}: no per-frame timestamps file")
            else:
                n_times = max(0, len(times.read_text().splitlines()) - 1)
                status = "PASS" if abs(n_times - len(stamps)) <= 2 else "WARN"
                sec.note(status, f"{video.name}: {video.stat().st_size / 1e6:.0f} MB, {n_times} timestamped frames "
                                 f"(tracking has {len(stamps)})")
        else:
            sec.note("WARN", f"{tracking.name}: no video recorded")

    # --- pulls ---------------------------------------------------------------------------
    sec = report.section("pulls")
    hidden = []
    for index in directions:
        path = apple_dir / "pulls" / f"d{index:02d}_robot.parquet"
        if not path.exists():
            sec.note("FAIL", f"d{index:02d}: not recorded")
            redo("grasp", "pulls")
            continue
        rows = _data_rows(path)
        meta = _metadata(path)
        ts = rows["timestamp"].to_numpy(float)
        rate = len(ts) / max(ts.max() - ts.min(), 1e-9)
        groups = rows.groupby(["hold_index", "phase"]).size()
        problems = []
        missing_keys = [k for k in ROBOT_TOP_LEVEL_KEYS if k not in meta]
        if missing_keys:
            problems.append(f"metadata missing {missing_keys}")
        if not 900 <= rate <= 1100:
            problems.append(f"rate {rate:.0f} Hz")
        expected_holds = int((meta.get("dump") or {}).get("n_holds") or stops)
        if len(groups) != 2 * expected_holds:
            problems.append(f"{len(groups)} hold/phase segments (expected {2 * expected_holds})")
        coverage = {}
        complete = 0
        for tracking, frame, stamps in tracking_frames:
            inside = frame[(frame["timestamp"] >= ts.min()) & (frame["timestamp"] <= ts.max())]
            if inside.empty:
                continue
            ok = np.isfinite(inside[["x", "y", "z"]].to_numpy(float)).all(axis=1)
            seen = inside[ok].groupby("timestamp")["name"].apply(set)
            complete += int(sum(set(TRACKERS) <= names for names in seen))
            n_frames = inside["timestamp"].nunique()
            for name in TRACKERS:
                coverage[name] = int(inside[ok & (inside["name"] == name)]["timestamp"].nunique()) / max(n_frames, 1)
        if complete == 0 and not tracking_frames:
            problems.append("no tracking to align with")
        elif complete == 0:
            hidden.append(f"d{index:02d}")
            problems.append("no camera frame with all 3 tags during the pull ("
                            + ", ".join(f"{k} {v * 100:.0f}%" for k, v in coverage.items()) + ")")
        status = "FAIL" if (missing_keys or complete == 0) else ("WARN" if problems else "PASS")
        sec.note(status, f"d{index:02d}: {len(rows)} rows, {rate:.0f} Hz, {complete} complete camera frames"
                         + (f"; {'; '.join(problems)}" if problems else ""))
    for aborted in sorted((apple_dir / "pulls").glob("*.aborted-*.parquet")):
        sec.note("PASS", f"kept aborted partial {aborted.name} (not used)")
    if hidden:
        report.redo.append(f"tags hidden during {', '.join(hidden)} (these pulls cannot be compiled): keep all 3 "
                           "tags visible with the apple held, then")
        redo("grasp", "pulls")

    # --- baselines -------------------------------------------------------------------
    sec = report.section("baselines")
    missing_baselines = []
    for index in directions:
        robot = apple_dir / "pulls" / f"d{index:02d}_robot.parquet"
        base = apple_dir / "baseline" / f"d{index:02d}_baseline.parquet"
        if not robot.exists():
            continue
        if not base.exists():
            missing_baselines.append(f"d{index:02d}")
            continue
        meta = _metadata(base)
        pull_span = np.ptp(_data_rows(robot)["timestamp"].to_numpy(float))
        base_span = np.ptp(_data_rows(base)["timestamp"].to_numpy(float))
        notes = []
        status = "PASS"
        if abs(base_span - pull_span) > 0.05 * pull_span:
            status = "WARN"
            notes.append(f"duration {base_span:.2f}s vs pull {pull_span:.2f}s")
        filt = meta.get("baseline_replay_filter")
        if not filt or filt.get("type") == "none":
            status = "WARN"
            notes.append("replayed the raw jittery velocities (recorded before the low-pass fix)")
        sec.note(status, f"d{index:02d}: start={meta.get('baseline_start_method')}"
                         + (f"; {'; '.join(notes)}" if notes else ""))
    if missing_baselines:
        sec.note("FAIL", f"missing for {', '.join(missing_baselines)}")
        redo("baseline")

    # --- parts + compiled ---------------------------------------------------------------
    sec = report.section("parts + compiled")
    parts = apple.get("parts")
    if not parts:
        sec.note("FAIL", "no measured parts")
        redo("measurements")
    else:
        try:
            from real_robot_exps.compile_static_sysid import _part_radii_from_parts

            radii = _part_radii_from_parts(parts)
            sec.note("PASS", "radii " + ", ".join(f"{k} {v * 1000:.1f} mm" for k, v in radii.items()))
        except Exception as exc:
            sec.note("FAIL", str(exc))
            redo("measurements")
        # Physically plausible? A typo accepted at the range prompt shows up here,
        # most clearly in the density derived from mass and dimensions.
        plausible = {
            "apple": {"radius_m": (0.015, 0.06), "density_kg_m3": (500.0, 1200.0)},
            "spur": {"radius_m": (0.0005, 0.015), "density_kg_m3": (300.0, 1500.0)},
            "stem": {"radius_m": (0.0003, 0.004), "density_kg_m3": (300.0, 1500.0)},
            "primary": {"radius_m": (0.003, 0.1)},
        }
        implausible = []
        for part, limits in plausible.items():
            for key, (lo, hi) in limits.items():
                value = (parts.get(part) or {}).get(key)
                if value is None or (part != "apple" and key == "density_kg_m3"
                                     and (parts.get(part) or {}).get("density_source", "").startswith("default")):
                    continue
                if not lo <= float(value) <= hi:
                    shown = f"{value * 1000:.1f} mm" if key == "radius_m" else f"{value:.0f} kg/m^3"
                    implausible.append(f"{part} {key.split('_')[0]} {shown}")
        if implausible:
            sec.note("WARN", "implausible measurements (typo?): " + "; ".join(implausible))
            redo("measurements")
    compiled = sorted((apple_dir / "compiled").glob("d[0-9][0-9].parquet"))
    if len(compiled) == len(directions):
        sec.note("PASS", f"{len(compiled)} compiled directions")
    else:
        sec.note("WARN", f"{len(compiled)}/{len(directions)} directions compiled (run --compile at home)")
    return report
