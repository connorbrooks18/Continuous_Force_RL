import json
import tempfile
import unittest
from pathlib import Path

from real_robot_exps.metadata_cache import (
    load_pre_grasp_metadata_cache,
    write_pre_grasp_metadata_cache,
)


class MetadataCacheTest(unittest.TestCase):
    def test_round_trip_preserves_raw_pre_grasp_geometry(self):
        structure = {
            "name": "orchard_a",
            "angles_source": "lengthened_state_check",
            "geometry_source": "catalog_entry_selected_by_structure_index",
            "parts": {
                "primary": {"shape": "cylinder", "length_m": 1.0, "radius_m": 0.1, "density_kg_m3": 2.0},
                "spur": {"shape": "cylinder", "length_m": 0.2, "radius_m": 0.02, "density_kg_m3": 3.0},
            },
        }
        pre_grasp_geometry = {
            "structure_index": 0,
            "structure_name": "orchard_a",
            "angles_source": "lengthened_state_check",
            "geometry_source": "catalog_entry_selected_by_structure_index",
            "note": "Manual structure catalog plus a lengthened camera snapshot for angle/length estimation.",
            "parts": {
                "primary": {
                    "shape": "cylinder",
                    "length_m": 1.0,
                    "radius_m": 0.1,
                    "density_kg_m3": 2.0,
                    "connection_rpy_deg": [0.0, 0.0, 0.0],
                    "connection_source": "catalog",
                },
                "spur": {
                    "shape": "cylinder",
                    "length_m": 0.2,
                    "radius_m": 0.02,
                    "density_kg_m3": 3.0,
                    "connection_rpy_deg": [1.0, 2.0, 3.0],
                    "connection_source": "lengthened_snapshot",
                },
            },
            "snapshot": {"apple_pos": [1.0, 2.0, 3.0]},
            "settled_snapshot": {},
            "under_gravity_snapshot": {},
            "lengthened_snapshot": {"apple_pos": [1.0, 2.0, 3.0]},
        }

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "metadata_cache.json"
            written = write_pre_grasp_metadata_cache(
                cache_path,
                structure_index=0,
                structure=structure,
                pre_grasp_geometry=pre_grasp_geometry,
            )

            with cache_path.open("r", encoding="utf-8") as f:
                on_disk = json.load(f)

            self.assertEqual(on_disk, pre_grasp_geometry)
            self.assertEqual(written, pre_grasp_geometry)

            loaded = load_pre_grasp_metadata_cache(
                cache_path,
                structure_index=0,
                structure=structure,
            )
            self.assertEqual(loaded, pre_grasp_geometry)

    def test_rejects_cache_for_different_structure_catalog(self):
        structure = {
            "name": "orchard_a",
            "angles_source": "lengthened_state_check",
            "geometry_source": "catalog_entry_selected_by_structure_index",
            "parts": {
                "primary": {"shape": "cylinder", "length_m": 1.0, "radius_m": 0.1, "density_kg_m3": 2.0},
            },
        }
        other_structure = {
            "name": "orchard_a",
            "angles_source": "lengthened_state_check",
            "geometry_source": "catalog_entry_selected_by_structure_index",
            "parts": {
                "primary": {"shape": "cylinder", "length_m": 1.0, "radius_m": 0.2, "density_kg_m3": 2.0},
            },
        }
        pre_grasp_geometry = {
            "structure_index": 0,
            "structure_name": "orchard_a",
            "angles_source": "lengthened_state_check",
            "geometry_source": "catalog_entry_selected_by_structure_index",
            "parts": {
                "primary": {
                    "shape": "cylinder",
                    "length_m": 1.0,
                    "radius_m": 0.1,
                    "density_kg_m3": 2.0,
                    "connection_rpy_deg": [0.0, 0.0, 0.0],
                    "connection_source": "catalog",
                },
            },
            "snapshot": {},
            "settled_snapshot": {},
            "under_gravity_snapshot": {},
            "lengthened_snapshot": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "metadata_cache.json"
            write_pre_grasp_metadata_cache(
                cache_path,
                structure_index=0,
                structure=structure,
                pre_grasp_geometry=pre_grasp_geometry,
            )

            self.assertIsNone(
                load_pre_grasp_metadata_cache(
                    cache_path,
                    structure_index=0,
                    structure=other_structure,
                )
            )


if __name__ == "__main__":
    unittest.main()
