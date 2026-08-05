import ast
import importlib.util
import unittest
from pathlib import Path


class RequirementsTests(unittest.TestCase):
    def test_requirements_match_product_dependencies_and_no_web_ui_imports_remain(self):
        lines = [
            line.strip() for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        names = {line.split("<")[0].split(">")[0].split("=")[0].lower() for line in lines}
        self.assertEqual(names, {"numpy", "pandas", "scipy", "pyyaml", "pyside6", "pyqtgraph"})
        for module in ("numpy", "pandas", "scipy", "yaml", "PySide6", "pyqtgraph"):
            self.assertIsNotNone(importlib.util.find_spec(module), module)

        imported = set()
        for root in ("fusion", "realtime", "visualization", "tools", "tests"):
            for path in Path(root).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.add(node.module.split(".")[0])
        self.assertNotIn("streamlit", imported)
        self.assertNotIn("plotly", imported)


if __name__ == "__main__":
    unittest.main()
