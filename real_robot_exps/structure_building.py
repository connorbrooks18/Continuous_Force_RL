"""Helpers for choosing or building a structure entry for the runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


InputFn = Callable[[str], str]
PrintFn = Callable[..., None]


DEFAULT_PRIMARY_PART = {
    "shape": "cylinder",
    "length_m": 0.827,
    "radius_m": 0.0125,
    "density_kg_m3": 660,
}
DEFAULT_SPUR_PART = {
    "shape": "cylinder",
    "length_m": 0.13,
    "radius_m": 0.0025,
    "density_kg_m3": 1200,
}
DEFAULT_STEM_PART = {
    "shape": "cylinder",
    "length_m": 0.005,
    "radius_m": 0.0005,
    "density_kg_m3": 1000,
}
DEFAULT_APPLE_PART = {
    "shape": "sphere",
    "length_m": 0.07,
    "radius_m": 0.035,
    "density_kg_m3": 650,
}

SPUR_STIFFNESS_ALIASES = {
    "0": (0, "low"),
    "l": (0, "low"),
    "low": (0, "low"),
    "1": (1, "medium"),
    "m": (1, "medium"),
    "med": (1, "medium"),
    "medium": (1, "medium"),
    "2": (2, "high"),
    "h": (2, "high"),
    "high": (2, "high"),
}
SPUR_LENGTH_ALIASES = {
    "0": (0, "short"),
    "short": (0, "short"),
    "1": (1, "medium"),
    "medium": (1, "medium"),
    "2": (2, "long"),
    "long": (2, "long"),
}
SPUR_LENGTH_TO_METERS = {
    0: 0.085,
    1: 0.13,
    2: 0.175,
}
SPUR_RADIUS_BY_STIFFNESS_LEVEL = {
    0: 0.0020,
    1: 0.0025,
    2: 0.0030,
}
STEM_ANGLE_CHOICES = {0, 30, 45, 60}
SPUR_ANGLE_CHOICES = {45, 60, 75, 90}


def load_structure_catalog(path: Path) -> list[dict[str, Any]]:
    """Load a structure catalog from JSON and return the structure list."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        structures = payload.get("structures", [])
    else:
        structures = payload
    if not isinstance(structures, list):
        raise ValueError(f"{path} does not contain a structure list")
    return structures


def _prompt_until_valid(
    prompt: str,
    parser: Callable[[str], Any],
    *,
    input_fn: InputFn,
    print_fn: PrintFn,
) -> Any:
    while True:
        raw = input_fn(prompt).strip()
        try:
            return parser(raw)
        except ValueError as exc:
            print_fn(f"[WARN] {exc}")


def _parse_int_choice(raw: str, *, valid: set[int], label: str) -> int:
    value = int(raw)
    if value not in valid:
        choices = ", ".join(str(choice) for choice in sorted(valid))
        raise ValueError(f"{label} must be one of: {choices}")
    return value


def _parse_stiffness(raw: str) -> tuple[int, str]:
    key = raw.lower()
    if key not in SPUR_STIFFNESS_ALIASES:
        raise ValueError("spur stiffness must be 0/1/2 or l/m/h")
    return SPUR_STIFFNESS_ALIASES[key]


def _parse_spur_length(raw: str) -> tuple[int, str]:
    key = raw.lower()
    if key not in SPUR_LENGTH_ALIASES:
        raise ValueError("spur length must be short/medium/long or 0/1/2")
    return SPUR_LENGTH_ALIASES[key]


def _build_manual_structure(
    *,
    apple_number: int,
    spur_stiffness: tuple[int, str],
    spur_length: tuple[int, str],
    stem_angle_deg: int,
    spur_angle_deg: int,
) -> dict[str, Any]:
    stiffness_level, stiffness_label = spur_stiffness
    spur_length_level, spur_length_label = spur_length
    apple_radius_m = round(0.035 + 0.0025 * (apple_number - 2), 6)
    if apple_radius_m < 0.025:
        apple_radius_m = 0.025

    spur_part = dict(DEFAULT_SPUR_PART)
    spur_part.update(
        {
            "length_m": SPUR_LENGTH_TO_METERS[spur_length_level],
            "radius_m": SPUR_RADIUS_BY_STIFFNESS_LEVEL[stiffness_level],
            "stiffness_level": stiffness_level,
            "stiffness_label": stiffness_label,
            "manual_selection": True,
            "manual_spur_angle_deg": spur_angle_deg,
            "connection_rpy_deg": [0.0, 0.0, float(spur_angle_deg)],
            "connection_source": "manual_selection",
        }
    )

    stem_part = dict(DEFAULT_STEM_PART)
    stem_part.update(
        {
            "manual_selection": True,
            "manual_stem_angle_deg": stem_angle_deg,
            "connection_rpy_deg": [0.0, float(stem_angle_deg), 0.0],
            "connection_source": "manual_selection",
        }
    )

    apple_part = dict(DEFAULT_APPLE_PART)
    apple_part.update(
        {
            "radius_m": apple_radius_m,
            "manual_selection": True,
            "apple_number": apple_number,
            "connection_rpy_deg": [0.0, 0.0, 0.0],
            "connection_source": "manual_selection",
        }
    )

    structure_name = (
        f"apple {apple_number} spur {stiffness_label} {spur_length_label} "
        f"stem {stem_angle_deg} spur {spur_angle_deg}"
    )
    return {
        "name": structure_name,
        "notes": (
            f"Manual structure entry for apple #{apple_number} with "
            f"spur stiffness={stiffness_label}, spur length={spur_length_label}, "
            f"stem angle={stem_angle_deg}, spur angle={spur_angle_deg}."
        ),
        "angles_source": "manual_runner_entry",
        "geometry_source": "manual_runner_entry",
        "manual_selection": {
            "apple_number": apple_number,
            "spur_stiffness": stiffness_level,
            "spur_stiffness_label": stiffness_label,
            "spur_length": spur_length_label,
            "spur_length_level": spur_length_level,
            "stem_angle_deg": stem_angle_deg,
            "spur_angle_deg": spur_angle_deg,
        },
        "parts": {
            "primary": dict(DEFAULT_PRIMARY_PART),
            "spur": spur_part,
            "stem": stem_part,
            "apple": apple_part,
        },
    }


def prompt_for_structure(
    structures: list[dict[str, Any]],
    *,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> tuple[int, dict[str, Any]]:
    """Prompt for a catalog selection or manual structure build."""
    if not structures:
        raise ValueError("No structures available to select")

    while True:
        print_fn("\nAvailable structures:")
        for idx, structure in enumerate(structures):
            print_fn(f"  {idx}: {structure.get('name', f'structure_{idx:02d}')}")
        print_fn("  n/no: build a new structure from prompted details")

        selected = input_fn("Structure index [0]: ").strip().lower()
        if selected in {"n", "no"}:
            apple_number = _prompt_until_valid(
                "Apple #: ",
                lambda raw: int(raw),
                input_fn=input_fn,
                print_fn=print_fn,
            )
            spur_stiffness = _prompt_until_valid(
                "Spur stiffness (0/1/2 or l/m/h): ",
                _parse_stiffness,
                input_fn=input_fn,
                print_fn=print_fn,
            )
            spur_length = _prompt_until_valid(
                "Spur length (short/medium/long or 0/1/2): ",
                _parse_spur_length,
                input_fn=input_fn,
                print_fn=print_fn,
            )
            stem_angle_deg = _prompt_until_valid(
                "Stem angle (0, 30, 45, 60): ",
                lambda raw: _parse_int_choice(raw, valid=STEM_ANGLE_CHOICES, label="stem angle"),
                input_fn=input_fn,
                print_fn=print_fn,
            )
            spur_angle_deg = _prompt_until_valid(
                "Spur angle (45, 60, 75, 90): ",
                lambda raw: _parse_int_choice(raw, valid=SPUR_ANGLE_CHOICES, label="spur angle"),
                input_fn=input_fn,
                print_fn=print_fn,
            )
            return len(structures), _build_manual_structure(
                apple_number=apple_number,
                spur_stiffness=spur_stiffness,
                spur_length=spur_length,
                stem_angle_deg=stem_angle_deg,
                spur_angle_deg=spur_angle_deg,
            )

        if not selected:
            return 0, structures[0]

        try:
            structure_index = int(selected)
        except ValueError:
            print_fn(f"[WARN] '{selected}' is not a valid structure index, 'n', or 'no'")
            continue

        if 0 <= structure_index < len(structures):
            return structure_index, structures[structure_index]
        print_fn(f"[WARN] Structure index must be in [0, {len(structures)})")
