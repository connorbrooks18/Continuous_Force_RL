"""Config helpers shared by the field-session scripts."""

from __future__ import annotations

from typing import Any


def parse_override_value(value_str: str) -> Any:
    """Same typing rules as apple_pullto_static --override: int, float, bool, else str."""
    for cast in (int, float):
        try:
            return cast(value_str)
        except ValueError:
            pass
    if value_str.lower() in {"true", "false"}:
        return value_str.lower() == "true"
    return value_str


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``key.path=value`` overrides in place and return the config."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be 'key=value', got: {override}")
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")
        parent = config
        for key in keys[:-1]:
            parent = parent.setdefault(key, {})
        parent[keys[-1]] = parse_override_value(value_str)
    return config
