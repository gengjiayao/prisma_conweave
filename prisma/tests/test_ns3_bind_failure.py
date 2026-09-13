import unittest
from unittest import mock
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ns3_model.ns3env import Ns3ZmqBridge
import zmq

class BindFailureTest(unittest.TestCase):
    def test_socket_bind_failure_raises_instead_of_exiting_successfully(self):
        with mock.patch('ns3_model.ns3env.zmq.Context') as context:
            context.return_value.socket.return_value.bind.side_effect=zmq.ZMQError('address in use')
            with self.assertRaises(RuntimeError):
                Ns3ZmqBridge(port=29999,startSim=False)

if __name__=='__main__':unittest.main()
