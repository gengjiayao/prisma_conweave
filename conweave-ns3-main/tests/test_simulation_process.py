import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor


SPEC = importlib.util.spec_from_file_location(
    "simulation_process", Path(__file__).resolve().parents[1] / "simulation_process.py"
)
PROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS)


class SimulatorExecutionTest(unittest.TestCase):
    def test_output_id_collision_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '42').mkdir()
            marker = root / '42' / 'results'
            marker.write_text('preserve')
            with mock.patch.object(PROCESS.secrets, 'randbelow', side_effect=[42, 43]):
                path = PROCESS.create_run_directory(root)
            self.assertEqual(path, root / '43')
            self.assertTrue(path.is_dir())
            self.assertEqual(marker.read_text(), 'preserve')

    def test_concurrent_output_reservations_are_unique(self):
        with tempfile.TemporaryDirectory() as folder:
            with ThreadPoolExecutor(max_workers=24) as pool:
                paths = list(pool.map(PROCESS.create_run_directory, [folder] * 96))
            self.assertEqual(len(set(paths)), 96)
            self.assertTrue(all(path.is_dir() for path in paths))

    def test_failing_simulator_propagates_status_and_preserves_log(self):
        with tempfile.TemporaryDirectory(prefix="simulator space ") as folder:
            root = Path(folder)
            config = root / "failure case.py"
            config.write_text("print('invalid routing state', flush=True)\nraise SystemExit(17)\n")
            with self.assertRaises(subprocess.CalledProcessError) as error:
                PROCESS.execute_simulator(config, root / "output.log",
                                          binary=sys.executable, cwd=root)
            self.assertEqual(error.exception.returncode, 17)
            self.assertIn("invalid routing state", (root / "output.log").read_text())

    def test_success_preserves_config_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="simulator space ") as folder:
            root = Path(folder)
            config = root / "successful run.py"
            config.write_text("print('completed all flows')\n")
            command = PROCESS.execute_simulator(config, root / "output.log",
                                                binary=sys.executable, cwd=root)
            self.assertEqual(command[-1], str(config))
            self.assertEqual((root / "output.log").read_text().strip(), "completed all flows")

    def test_missing_binary_fails_before_starting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                PROCESS.execute_simulator(root / "config.txt", root / "output.log",
                                          binary=root / "missing", cwd=root)
            self.assertFalse((root / "output.log").exists())


if __name__ == "__main__":
    unittest.main()
