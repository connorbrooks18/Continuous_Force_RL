from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

import real_robot_exps.structure_building as sb


class StructureBuildingTest(unittest.TestCase):
    def test_load_structure_catalog_reads_wrapped_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "structures.json"
            path.write_text(
                """
                {
                  "structures": [
                    {"name": "orchard_a"},
                    {"name": "orchard_b"}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            structures = sb.load_structure_catalog(path)

        self.assertEqual([entry["name"] for entry in structures], ["orchard_a", "orchard_b"])

    def test_prompt_for_structure_builds_manual_entry(self):
        responses = iter([
            "no",
            "3",
            "h",
            "long",
            "45",
            "90",
        ])
        printed = []

        def fake_input(prompt: str) -> str:
            printed.append(prompt)
            return next(responses)

        def fake_print(*args, **kwargs):
            printed.append(" ".join(str(arg) for arg in args))

        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "structures.json"
            catalog_path.write_text(
                json.dumps({"structures": [{"name": "default_template"}]}, indent=2),
                encoding="utf-8",
            )
            structures = sb.load_structure_catalog(catalog_path)
            structure_index, structure = sb.prompt_for_structure(
                structures,
                catalog_path=catalog_path,
                input_fn=fake_input,
                print_fn=fake_print,
            )

            self.assertEqual(structure_index, 1)
            self.assertEqual(structure["manual_selection"]["apple_number"], 3)
            self.assertEqual(structure["manual_selection"]["spur_stiffness"], 2)
            self.assertEqual(structure["manual_selection"]["spur_length"], "long")
            self.assertEqual(structure["parts"]["primary"]["radius_m"], 0.0125)
            self.assertEqual(structure["parts"]["spur"]["radius_m"], 0.0059)
            self.assertAlmostEqual(structure["parts"]["spur"]["mass_kg"], 0.01574767299909034)
            self.assertEqual(structure["parts"]["apple"]["radius_m"], 0.04)
            self.assertEqual(structure["parts"]["stem"]["connection_rpy_deg"], [0.0, 45.0, 0.0])
            self.assertEqual(structure["parts"]["spur"]["connection_rpy_deg"], [0.0, 90.0, 0.0])
            self.assertIn("n/no: build a new structure", "\n".join(printed))

            on_disk = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(len(on_disk["structures"]), 2)
            self.assertEqual(on_disk["structures"][1]["name"], structure["name"])
            self.assertEqual(on_disk["structures"][1]["parts"]["apple"]["radius_m"], 0.04)
            self.assertAlmostEqual(on_disk["structures"][1]["parts"]["spur"]["mass_kg"], 0.01574767299909034)

    def test_prompt_for_structure_uses_longer_stems_for_apples_two_and_four(self):
        for apple_number in (2, 4):
            responses = iter([
                "no",
                str(apple_number),
                "l",
                "short",
                "30",
                "45",
            ])
            printed = []

            def fake_input(prompt: str) -> str:
                printed.append(prompt)
                return next(responses)

            def fake_print(*args, **kwargs):
                printed.append(" ".join(str(arg) for arg in args))

            with tempfile.TemporaryDirectory() as tmp:
                catalog_path = Path(tmp) / "structures.json"
                catalog_path.write_text(
                    json.dumps({"structures": [{"name": "default_template"}]}, indent=2),
                    encoding="utf-8",
                )
                structures = sb.load_structure_catalog(catalog_path)
                structure_index, structure = sb.prompt_for_structure(
                    structures,
                    catalog_path=catalog_path,
                    input_fn=fake_input,
                    print_fn=fake_print,
                )

            self.assertEqual(structure_index, 1)
            self.assertEqual(structure["manual_selection"]["apple_number"], apple_number)
            self.assertEqual(structure["parts"]["stem"]["length_m"], 0.015)
            self.assertAlmostEqual(structure["parts"]["spur"]["mass_kg"], 0.001884955592153876)
            if apple_number == 2:
                self.assertAlmostEqual(structure["parts"]["apple"]["density_kg_m3"], 749.7689897219757)
            else:
                self.assertAlmostEqual(structure["parts"]["apple"]["density_kg_m3"], 1059.3750899554282)
            self.assertIn("n/no: build a new structure", "\n".join(printed))

    def test_prompt_for_structure_keeps_default_stem_length_for_other_apples(self):
        responses = iter([
            "no",
            "3",
            "l",
            "short",
            "30",
            "45",
        ])
        printed = []

        def fake_input(prompt: str) -> str:
            printed.append(prompt)
            return next(responses)

        def fake_print(*args, **kwargs):
            printed.append(" ".join(str(arg) for arg in args))

        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "structures.json"
            catalog_path.write_text(
                json.dumps({"structures": [{"name": "default_template"}]}, indent=2),
                encoding="utf-8",
            )
            structures = sb.load_structure_catalog(catalog_path)
            _, structure = sb.prompt_for_structure(
                structures,
                catalog_path=catalog_path,
                input_fn=fake_input,
                print_fn=fake_print,
            )

        self.assertEqual(structure["parts"]["stem"]["length_m"], 0.005)
        self.assertAlmostEqual(structure["parts"]["spur"]["mass_kg"], 0.001884955592153876)
        self.assertIn("n/no: build a new structure", "\n".join(printed))

    def test_prompt_for_structure_prompts_and_persists_missing_spur_mass(self):
        responses = iter([
            "no",
            "3",
            "h",
            "long",
            "45",
            "90",
            "0.02",
        ])
        printed = []

        def fake_input(prompt: str) -> str:
            printed.append(prompt)
            return next(responses)

        def fake_print(*args, **kwargs):
            printed.append(" ".join(str(arg) for arg in args))

        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "structures.json"
            catalog_path.write_text(
                json.dumps({"structures": [{"name": "default_template"}]}, indent=2),
                encoding="utf-8",
            )
            constants_path = Path(tmp) / "structure_constants.json"
            constants = deepcopy(sb.STRUCTURE_CONSTANTS)
            constants["spur_mass_kg_by_stiffness_and_length"] = {}
            constants_path.write_text(json.dumps(constants, indent=2, sort_keys=True), encoding="utf-8")

            original_constants = sb.STRUCTURE_CONSTANTS
            original_constants_path = sb.STRUCTURE_CONSTANTS_PATH
            try:
                sb.STRUCTURE_CONSTANTS = constants
                sb.STRUCTURE_CONSTANTS_PATH = constants_path
                structures = sb.load_structure_catalog(catalog_path)
                _, structure = sb.prompt_for_structure(
                    structures,
                    catalog_path=catalog_path,
                    input_fn=fake_input,
                    print_fn=fake_print,
                )
            finally:
                sb.STRUCTURE_CONSTANTS = original_constants
                sb.STRUCTURE_CONSTANTS_PATH = original_constants_path

            self.assertAlmostEqual(structure["parts"]["spur"]["mass_kg"], 0.02)
            self.assertIn("Missing spur mass entry", "\n".join(printed))

            updated = json.loads(constants_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["spur_mass_kg_by_stiffness_and_length"]["2"]["2"], 0.02)


if __name__ == "__main__":
    unittest.main()
