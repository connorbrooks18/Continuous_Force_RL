"""Helpers for choosing or building a structure entry for the runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


InputFn = Callable[[str], str]
PrintFn = Callable[..., None]


STRUCTURE_CONSTANTS_PATH = Path(__file__).with_name("structure_constants.json")


def _load_structure_constants(path: Path = STRUCTURE_CONSTANTS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _int_keyed_mapping(raw: dict[str, Any]) -> dict[int, Any]:
    return {int(key): value for key, value in raw.items()}


def _nested_int_keyed_mapping(raw: dict[str, Any]) -> dict[int, dict[int, Any]]:
    return {int(key): _int_keyed_mapping(value) for key, value in raw.items()}


STRUCTURE_CONSTANTS = _load_structure_constants()

DEFAULT_PRIMARY_PART = dict(STRUCTURE_CONSTANTS["default_parts"]["primary"])
DEFAULT_SPUR_PART = dict(STRUCTURE_CONSTANTS["default_parts"]["spur"])
DEFAULT_STEM_PART = dict(STRUCTURE_CONSTANTS["default_parts"]["stem"])
DEFAULT_APPLE_PART = dict(STRUCTURE_CONSTANTS["default_parts"]["apple"])

SPUR_STIFFNESS_ALIASES = {
    key: (int(value[0]), str(value[1]))
    for key, value in STRUCTURE_CONSTANTS["spur_stiffness_aliases"].items()
}
SPUR_LENGTH_ALIASES = {
    key: (int(value[0]), str(value[1]))
    for key, value in STRUCTURE_CONSTANTS["spur_length_aliases"].items()
}
SPUR_LENGTH_TO_METERS = {
    int(key): float(value) for key, value in STRUCTURE_CONSTANTS["spur_length_to_meters"].items()
}
SPUR_RADIUS_BY_STIFFNESS_LEVEL = {
    int(key): float(value) for key, value in STRUCTURE_CONSTANTS["spur_radius_by_stiffness_level"].items()
}
APPLE_RADIUS_BY_NUMBER = {
    int(key): float(value) for key, value in STRUCTURE_CONSTANTS["apple_radius_by_number"].items()
}
APPLE_DENSITY_BY_NUMBER = {
    int(key): float(value) for key, value in STRUCTURE_CONSTANTS["apple_density_kg_m3_by_number"].items()
}
STEM_ANGLE_CHOICES = {int(choice) for choice in STRUCTURE_CONSTANTS["stem_angle_choices"]}
SPUR_ANGLE_CHOICES = {int(choice) for choice in STRUCTURE_CONSTANTS["spur_angle_choices"]}
STEM_LENGTH_BY_APPLE_NUMBER = {
    int(key): float(value) for key, value in STRUCTURE_CONSTANTS["stem_length_by_apple_number"].items()
}


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


def append_structure_to_catalog(
    path: Path,
    structure: dict[str, Any],
    *,
    structures: list[dict[str, Any]] | None = None,
) -> int:
    """Append a structure to ``path`` and return the saved structure index."""
    if structures is None:
        structures = load_structure_catalog(path)
    structures.append(structure)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        catalog = payload.get("structures", [])
        if not isinstance(catalog, list):
            raise ValueError(f"{path} does not contain a structure list")
        catalog.append(structure)
        payload["structures"] = catalog
    elif isinstance(payload, list):
        payload.append(structure)
    else:
        raise ValueError(f"{path} does not contain a JSON array or object catalog")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return len(structures) - 1


def _save_structure_constants(path: Path | None = None) -> None:
    if path is None:
        path = STRUCTURE_CONSTANTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(STRUCTURE_CONSTANTS, f, indent=2, sort_keys=True)


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


def _apple_radius_from_number(apple_number: int) -> float:
    if apple_number not in APPLE_RADIUS_BY_NUMBER:
        choices = ", ".join(str(choice) for choice in sorted(APPLE_RADIUS_BY_NUMBER))
        raise ValueError(f"apple # must be one of: {choices}")
    return APPLE_RADIUS_BY_NUMBER[apple_number]


def _apple_density_from_number(apple_number: int) -> float:
    return APPLE_DENSITY_BY_NUMBER.get(apple_number, DEFAULT_APPLE_PART["density_kg_m3"])


def _stem_length_from_number(apple_number: int) -> float:
    return STEM_LENGTH_BY_APPLE_NUMBER.get(apple_number, DEFAULT_STEM_PART["length_m"])


def _stem_length_table() -> dict[str, Any]:
    length_table = STRUCTURE_CONSTANTS.get("stem_length_by_apple_number")
    if not isinstance(length_table, dict):
        length_table = {}
        STRUCTURE_CONSTANTS["stem_length_by_apple_number"] = length_table
    return length_table


def _stem_length_from_apple_number(
    apple_number: int,
    *,
    input_fn: InputFn,
    print_fn: PrintFn,
) -> float:
    length_table = _stem_length_table()
    apple_key = str(apple_number)
    stored_length = length_table.get(apple_key)
    if stored_length is not None:
        return float(stored_length)

    print_fn(f"[INFO] Missing stem length entry for apple #{apple_number}.")
    stem_length_m = _prompt_until_valid(
        "Measured stem length in meters: ",
        lambda raw: float(raw),
        input_fn=input_fn,
        print_fn=print_fn,
    )
    length_table[apple_key] = float(stem_length_m)
    _save_structure_constants()
    print_fn(
        f"Saved stem length {float(stem_length_m)} m to {STRUCTURE_CONSTANTS_PATH} "
        f"for apple #{apple_number}"
    )
    return float(stem_length_m)


def _spur_mass_table() -> dict[str, Any]:
    mass_table = STRUCTURE_CONSTANTS.get("spur_mass_kg_by_stiffness_and_length")
    if not isinstance(mass_table, dict):
        mass_table = {}
        STRUCTURE_CONSTANTS["spur_mass_kg_by_stiffness_and_length"] = mass_table
    return mass_table


def _spur_mass_from_parameters(
    *,
    stiffness_level: int,
    stiffness_label: str,
    length_level: int,
    length_label: str,
    input_fn: InputFn,
    print_fn: PrintFn,
) -> float:
    mass_table = _spur_mass_table()
    stiffness_key = str(stiffness_level)
    length_key = str(length_level)
    stored_mass = mass_table.get(stiffness_key, {}).get(length_key)
    if stored_mass is not None:
        return float(stored_mass)

    print_fn(
        "[INFO] Missing spur mass entry for "
        f"stiffness={stiffness_label} ({stiffness_level}), length={length_label} ({length_level})."
    )
    mass_kg = _prompt_until_valid(
        "Measured spur mass in kg: ",
        lambda raw: float(raw),
        input_fn=input_fn,
        print_fn=print_fn,
    )
    mass_table.setdefault(stiffness_key, {})[length_key] = float(mass_kg)
    _save_structure_constants()
    print_fn(
        f"Saved spur mass {float(mass_kg)} kg to {STRUCTURE_CONSTANTS_PATH} "
        f"for stiffness={stiffness_label}, length={length_label}"
    )
    return float(mass_kg)


def _build_manual_structure(
    *,
    apple_number: int,
    spur_stiffness: tuple[int, str],
    spur_length: tuple[int, str],
    stem_angle_deg: int,
    spur_angle_deg: int,
    input_fn: InputFn,
    print_fn: PrintFn,
) -> dict[str, Any]:
    stiffness_level, stiffness_label = spur_stiffness
    spur_length_level, spur_length_label = spur_length
    apple_radius_m = _apple_radius_from_number(apple_number)

    spur_part = dict(DEFAULT_SPUR_PART)
    spur_part.update(
        {
            "length_m": SPUR_LENGTH_TO_METERS[spur_length_level],
            "radius_m": SPUR_RADIUS_BY_STIFFNESS_LEVEL[stiffness_level],
            "mass_kg": _spur_mass_from_parameters(
                stiffness_level=stiffness_level,
                stiffness_label=stiffness_label,
                length_level=spur_length_level,
                length_label=spur_length_label,
                input_fn=input_fn,
                print_fn=print_fn,
            ),
            "stiffness_level": stiffness_level,
            "stiffness_label": stiffness_label,
            "manual_selection": True,
            "manual_spur_angle_deg": spur_angle_deg,
            "connection_rpy_deg": [0.0, float(spur_angle_deg), 0.0],
            "connection_source": "manual_selection",
        }
    )

    stem_part = dict(DEFAULT_STEM_PART)
    stem_part.update(
        {
            "length_m": _stem_length_from_apple_number(
                apple_number,
                input_fn=input_fn,
                print_fn=print_fn,
            ),
            "manual_selection": True,
            "manual_stem_angle_deg": stem_angle_deg,
            "connection_rpy_deg": [0.0, float(stem_angle_deg), 0.0],
            "connection_source": "manual_selection",
        }
    )

    apple_part = dict(DEFAULT_APPLE_PART)
    apple_part.update(
        {
            "density_kg_m3": _apple_density_from_number(apple_number),
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
    catalog_path: Path | None = None,
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
                "Stem pitch (0, 30, 45, 60, 90): ",
                lambda raw: _parse_int_choice(raw, valid=STEM_ANGLE_CHOICES, label="stem angle"),
                input_fn=input_fn,
                print_fn=print_fn,
            )
            spur_angle_deg = _prompt_until_valid(
                "Spur pitch (45, 60, 75, 90): ",
                lambda raw: _parse_int_choice(raw, valid=SPUR_ANGLE_CHOICES, label="spur angle"),
                input_fn=input_fn,
                print_fn=print_fn,
            )
            manual_structure = _build_manual_structure(
                apple_number=apple_number,
                spur_stiffness=spur_stiffness,
                spur_length=spur_length,
                stem_angle_deg=stem_angle_deg,
                spur_angle_deg=spur_angle_deg,
                input_fn=input_fn,
                print_fn=print_fn,
            )
            if catalog_path is not None:
                structure_index = append_structure_to_catalog(
                    catalog_path,
                    manual_structure,
                    structures=structures,
                )
                print_fn(f"Saved manual structure to {catalog_path} at index {structure_index}")
                return structure_index, manual_structure
            structures.append(manual_structure)
            return len(structures) - 1, manual_structure

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
