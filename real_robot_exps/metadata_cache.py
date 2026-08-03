"""Helpers for caching measured pre-grasp metadata between runner invocations."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _structure_part_signature(part: dict[str, Any]) -> dict[str, Any]:
    signature = {
        "shape": part.get("shape"),
        "length_m": part.get("length_m"),
        "radius_m": part.get("radius_m"),
        "density_kg_m3": part.get("density_kg_m3"),
    }
    if "mass_kg" in part and signature["density_kg_m3"] is None:
        signature["density_kg_m3"] = part.get("mass_kg")
    return signature


def _structure_signature_from_structure(structure_index: int, structure: dict[str, Any]) -> dict[str, Any]:
    parts = structure.get("parts", {}) or {}
    return {
        "structure_index": int(structure_index),
        "structure_name": str(structure.get("name", f"structure_{int(structure_index):02d}")),
        "angles_source": str(structure.get("angles_source", "")),
        "geometry_source": str(structure.get("geometry_source", "")),
        "parts": {name: _structure_part_signature(dict(parts.get(name, {}) or {})) for name in sorted(parts)},
    }


def _structure_signature_from_pre_grasp_geometry(pre_grasp_geometry: dict[str, Any]) -> dict[str, Any]:
    parts = pre_grasp_geometry.get("parts", {}) or {}
    return {
        "structure_index": int(pre_grasp_geometry.get("structure_index", -1)),
        "structure_name": str(pre_grasp_geometry.get("structure_name", "")),
        "angles_source": str(pre_grasp_geometry.get("angles_source", "")),
        "geometry_source": str(pre_grasp_geometry.get("geometry_source", "")),
        "parts": {name: _structure_part_signature(dict(parts.get(name, {}) or {})) for name in sorted(parts)},
    }


def write_pre_grasp_metadata_cache(
    path: Path,
    *,
    structure_index: int,
    structure: dict[str, Any],
    pre_grasp_geometry: dict[str, Any],
) -> dict[str, Any]:
    """Write the current pre-grasp metadata cache and return the payload."""
    payload = deepcopy(pre_grasp_geometry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return payload


def load_pre_grasp_metadata_cache(
    path: Path,
    *,
    structure_index: int,
    structure: dict[str, Any],
) -> dict[str, Any] | None:
    """Load a matching pre-grasp cache or return ``None`` if it is unusable."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")

    if "pre_grasp_geometry" in payload and isinstance(payload["pre_grasp_geometry"], dict):
        payload = payload["pre_grasp_geometry"]

    expected_signature = _structure_signature_from_structure(structure_index, structure)
    cached_signature = _structure_signature_from_pre_grasp_geometry(payload)
    if cached_signature != expected_signature:
        return None
    return payload
