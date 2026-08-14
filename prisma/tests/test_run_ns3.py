import os
import sys
import tempfile
import unittest
from unittest import mock


PRISMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PRISMA_DIR not in sys.path:
    sys.path.insert(0, PRISMA_DIR)

from source.run_ns3 import run_ns3  # noqa: E402


class RunNs3ProcessContractTest(unittest.TestCase):
    def test_returns_owned_process_handle(self):
        with tempfile.TemporaryDirectory() as ns3_dir:
            open(os.path.join(ns3_dir, "waf"), "w").close()
            params = {
                "ns3_sim_path": ns3_dir,
                "cc": "dctcp",
                "lb": "fecmp",
                "pfc": 0,
                "irn": 1,
                "simul_time": 0.01,
                "buffer": 50,
                "netload": 70,
                "bw": 1,
                "topo": "test_topology",
                "cdf": "AliStorage2019",
                "enforce_win": 0,
                "sw_monitoring_interval": 1000000,
                "seed": 100,
                "traffic_seed": 200,
                "session_name": "unit",
                "basePort": 6555,
                "overlay_adjacency_matrix_path": "/tmp/overlay.txt",
                "index_to_switch_id_map_path": "/tmp/index.txt",
            }
            fake_process = mock.Mock(pid=1234)
            with mock.patch("subprocess.Popen", return_value=fake_process) as popen:
                result = run_ns3(params)
            self.assertIs(result, fake_process)
            command = popen.call_args.args[0]
            self.assertIn("--traffic_seed", command)
            self.assertEqual(command[command.index("--traffic_seed") + 1], "200")
            self.assertNotIn("--cwh_extra_reply_deadline", command)

    def test_omits_traffic_seed_for_legacy_runs(self):
        with tempfile.TemporaryDirectory() as ns3_dir:
            open(os.path.join(ns3_dir, "waf"), "w").close()
            params = {
                "ns3_sim_path": ns3_dir,
                "cc": "dctcp",
                "lb": "fecmp",
                "pfc": 0,
                "irn": 1,
                "simul_time": 0.01,
                "buffer": 50,
                "netload": 70,
                "bw": 1,
                "topo": "test_topology",
                "cdf": "AliStorage2019",
                "enforce_win": 0,
                "sw_monitoring_interval": 1000000,
                "seed": 100,
                "session_name": "unit",
                "basePort": 6555,
                "overlay_adjacency_matrix_path": "/tmp/overlay.txt",
                "index_to_switch_id_map_path": "/tmp/index.txt",
            }
            with mock.patch("subprocess.Popen", return_value=mock.Mock(pid=1234)) as popen:
                run_ns3(params)
            self.assertNotIn("--traffic_seed", popen.call_args.args[0])

    def test_forwards_explicit_conweave_timing_overrides(self):
        with tempfile.TemporaryDirectory() as ns3_dir:
            open(os.path.join(ns3_dir, "waf"), "w").close()
            params = {
                "ns3_sim_path": ns3_dir,
                "cc": "dcqcn",
                "lb": "conweave",
                "pfc": 0,
                "irn": 1,
                "simul_time": 0.01,
                "buffer": 50,
                "netload": 70,
                "bw": 1,
                "topo": "test_topology",
                "cdf": "AliStorage2019",
                "enforce_win": 0,
                "sw_monitoring_interval": 1000000,
                "seed": 100,
                "session_name": "unit",
                "basePort": 6555,
                "overlay_adjacency_matrix_path": "/tmp/overlay.txt",
                "index_to_switch_id_map_path": "/tmp/index.txt",
                "cwh_extra_reply_deadline": 400,
                "cwh_path_pause_time": 1600,
                "cwh_extra_voq_flush_time": 1600,
                "cwh_default_voq_waiting_time": 2000,
                "cwh_tx_expiry_time": 3000,
            }
            with mock.patch("subprocess.Popen", return_value=mock.Mock(pid=1234)) as popen:
                run_ns3(params)
            command = popen.call_args.args[0]
            expected = {
                "--cwh_extra_reply_deadline": "400",
                "--cwh_path_pause_time": "1600",
                "--cwh_extra_voq_flush_time": "1600",
                "--cwh_default_voq_waiting_time": "2000",
                "--cwh_tx_expiry_time": "3000",
            }
            for option, value in expected.items():
                self.assertIn(option, command)
                self.assertEqual(command[command.index(option) + 1], value)


if __name__ == "__main__":
    unittest.main()
