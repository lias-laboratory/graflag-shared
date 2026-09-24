"""graflag_runner declares psutil as its only dependency, so it has to import,
and write a results file, where numpy is not installed.

1.1.0 imported numpy unconditionally in serialization.py, and the package
failed to import at all without it. Each check runs in a fresh interpreter in
which `import numpy` raises ImportError.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

LIBS = Path(__file__).resolve().parents[2]


def run_without_numpy(code):
    """Run `code` with numpy made unimportable; return the completed process."""
    script = "import sys\nsys.modules['numpy'] = None\n" + textwrap.dedent(code)
    env = dict(os.environ, PYTHONPATH=str(LIBS))
    return subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=env)


class WithoutNumpy(unittest.TestCase):

    def test_the_package_imports(self):
        result = run_without_numpy("import graflag_runner\n")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plain_values_still_serialize(self):
        result = run_without_numpy('''
            import json
            from graflag_runner.serialization import json_default, sanitize
            clean, replaced = sanitize({"scores": [0.5, float("nan")], "tags": {"a"}})
            print(json.dumps(clean, default=json_default), replaced)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '{"scores": [0.5, null], "tags": ["a"]} 1')

    def test_a_results_file_is_written(self):
        with tempfile.TemporaryDirectory() as exp:
            result = run_without_numpy(f'''
                from graflag_runner import ResultWriter
                writer = ResultWriter({exp!r})
                writer.save_scores(result_type="NODE_ANOMALY_SCORES",
                                   scores=[0.1, 0.9], ground_truth=[0, 1])
                writer.finalize()
            ''')
            self.assertEqual(result.returncode, 0, result.stderr)
            written = json.loads(Path(exp, "results.json").read_text())
            self.assertEqual(written["scores"], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
