import tempfile
import unittest
from pathlib import Path

from real_robot_exps.structure_building import load_structure_catalog, prompt_for_structure


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

            structures = load_structure_catalog(path)

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

        structures = [{"name": "default_template"}]
        structure_index, structure = prompt_for_structure(
            structures,
            input_fn=fake_input,
            print_fn=fake_print,
        )

        self.assertEqual(structure_index, 1)
        self.assertEqual(structure["manual_selection"]["apple_number"], 3)
        self.assertEqual(structure["manual_selection"]["spur_stiffness"], 2)
        self.assertEqual(structure["manual_selection"]["spur_length"], "long")
        self.assertEqual(structure["parts"]["primary"]["radius_m"], 0.0125)
        self.assertEqual(structure["parts"]["spur"]["radius_m"], 0.003)
        self.assertEqual(structure["parts"]["apple"]["radius_m"], 0.0375)
        self.assertEqual(structure["parts"]["stem"]["connection_rpy_deg"], [0.0, 45.0, 0.0])
        self.assertEqual(structure["parts"]["spur"]["connection_rpy_deg"], [0.0, 0.0, 90.0])
        self.assertIn("n/no: build a new structure", "\n".join(printed))


if __name__ == "__main__":
    unittest.main()
