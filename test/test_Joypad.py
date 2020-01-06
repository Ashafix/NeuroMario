import unittest
from Joypad import Joypad
import numpy as np


class TestJoypad(unittest.TestCase):

    def test_joypad_to_array(self):

        # makes sure that input and output are converted identically
        for j in dir(Joypad):
            if not j.startswith('__') and not callable(getattr(Joypad, j)):
                if not isinstance(getattr(Joypad, j), bytes):
                    continue

                a = Joypad.joypad_to_array(getattr(Joypad, j))
                self.assertEqual(bytes(getattr(Joypad, j)),
                                 Joypad.array_to_joypad(a, player=0))
        # same test but only for player 1
        for j in dir(Joypad):
            if not j.startswith('__') and not callable(getattr(Joypad, j)):
                if not isinstance(getattr(Joypad, j), bytes):
                    continue

                a = Joypad.joypad_to_array(getattr(Joypad, j))[4:16]
                self.assertEqual(bytes(getattr(Joypad, j)),
                                 Joypad.array_to_joypad(a, player=1))

    def test_array_to_joypad(self):

        arr0 = np.zeros((12, ))
        arr1 = np.ones((12, ))

        # test all to all conversions
        self.assertEqual(Joypad.array_to_joypad(arr0),
                         Joypad.empty)
        self.assertEqual(Joypad.array_to_joypad(arr0, player=2),
                         Joypad.empty)
        self.assertEqual(Joypad.array_to_joypad(arr1),
                         Joypad.all_but)
        self.assertEqual(Joypad.array_to_joypad(arr0, threshold=-0.1),
                         Joypad.all_but)

        # test specific conversion
        arr0 = np.zeros((12,))
        arr0[0] = 1
        self.assertEqual(Joypad.array_to_joypad(arr0),
                         Joypad.up)
        arr0 = np.zeros((12,))
        arr0[11] = 1
        self.assertEqual(Joypad.array_to_joypad(arr0),
                         Joypad.R1)
        arr0 = np.zeros((12,))
        arr0[2] = 1
        self.assertEqual(Joypad.array_to_joypad(arr0),
                         Joypad.left)
        arr0 = np.zeros((12,))
        arr0[7] = 1
        self.assertEqual(Joypad.array_to_joypad(arr0),
                         Joypad.B)

    def test_flip_input(self):
        j = Joypad
        self.assertEqual(j.flip_input(j.down), j.up)
        self.assertEqual(j.flip_input(j.up), j.down)
        self.assertEqual(j.flip_input(j.right), j.left)
        self.assertEqual(j.flip_input(j.left), j.right)

        self.assertEqual(j.flip_input(j.left, left_right=False), j.left)
        self.assertEqual(j.flip_input(j.right, left_right=False), j.right)
        self.assertEqual(j.flip_input(j.down, down_up=False), j.down)
        self.assertEqual(j.flip_input(j.up, down_up=False), j.up)
        self.assertEqual(j.flip_input(b"|..|U.L.........|............|"), b"|..|.D.R........|............|")

        with self.assertRaises(TypeError):
            j.flip_input("|..|U.L.........|............|")
        with self.assertRaises(TypeError):
            j.flip_input(1)

        # TODO add cases for arrays


if __name__ == '__main__':
    unittest.main()
