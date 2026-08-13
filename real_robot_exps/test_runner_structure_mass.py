import unittest

from real_robot_exps.runner import _normalized_pre_grasp_geometry


class RunnerStructureMassTest(unittest.TestCase):
    def test_normalized_pre_grasp_geometry_adds_spur_mass(self):
        structure = {
            "name": "orchard_a",
            "angles_source": "lengthened_state_check",
            "geometry_source": "catalog_entry_selected_by_structure_index",
            "parts": {
                "primary": {"shape": "cylinder", "length_m": 1.0, "radius_m": 0.1, "density_kg_m3": 2.0},
                "spur": {"shape": "cylinder", "length_m": 0.12, "radius_m": 0.0059, "density_kg_m3": 1200},
                "stem": {"shape": "cylinder", "length_m": 0.01, "radius_m": 0.001, "density_kg_m3": 1000},
                "apple": {"shape": "sphere", "length_m": 0.07, "radius_m": 0.04, "density_kg_m3": 650},
            },
        }

        normalized = _normalized_pre_grasp_geometry(3, structure)

        self.assertAlmostEqual(normalized["parts"]["spur"]["mass_kg"], 0.01574767299909034)
        self.assertEqual(normalized["parts"]["spur"]["connection_source"], "catalog_or_lengthened_state_placeholder")


if __name__ == "__main__":
    unittest.main()
