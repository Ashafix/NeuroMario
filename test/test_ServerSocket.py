import unittest
from ServerSocket import ServerSocket


class TestServerSocket(unittest.TestCase):
    def test_init(self):
        s = ServerSocket()
        self.assertIsNotNone(s.ip)
        self.assertEqual(s.port, 0)

    def test_str(self):
        s = ServerSocket()
        s_str = str(s)
        self.assertIn('ip: {}'.format(s.ip), s_str)
        self.assertIn('port: {}'.format(s.port), s_str)

    def test_repr(self):
        s = ServerSocket()
        s_repr = s.__repr__()
        self.assertEqual(str(s), s_repr)


if __name__ == '__main__':
    unittest.main()

