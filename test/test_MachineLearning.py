import unittest
from MachineLearning import MachineLearning
import numpy as np
from PIL import Image


class TestMachineLearning(unittest.TestCase):
    def test_flip_image(self):
        img_array = np.array(Image.open('test/rotate.png'))

        ML = MachineLearning
        # trivial tests
        self.assertNotEqual(img_array.tolist(), ML.flip_image(img_array, 'up').tolist())
        self.assertNotEqual(img_array.tolist(), ML.flip_image(img_array, 'down').tolist())
        self.assertNotEqual(img_array.tolist(), ML.flip_image(img_array, 'left').tolist())
        self.assertNotEqual(img_array.tolist(), ML.flip_image(img_array, 'right').tolist())

        # should not matter which way we flip it
        self.assertEqual(ML.flip_image(img_array, 'down').tolist(), ML.flip_image(img_array, 'up').tolist())
        self.assertEqual(ML.flip_image(img_array, 'up').tolist(), ML.flip_image(img_array, 'down').tolist())
        self.assertEqual(ML.flip_image(img_array, 'right').tolist(), ML.flip_image(img_array, 'left').tolist())
        self.assertEqual(ML.flip_image(img_array, 'left').tolist(), ML.flip_image(img_array, 'right').tolist())
        # check if numbers and strings match
        self.assertEqual(ML.flip_image(img_array, 0).tolist(), ML.flip_image(img_array, 'up').tolist())
        self.assertEqual(ML.flip_image(img_array, 1).tolist(), ML.flip_image(img_array, 'down').tolist())
        self.assertEqual(ML.flip_image(img_array, 2).tolist(), ML.flip_image(img_array, 'left').tolist())
        self.assertEqual(ML.flip_image(img_array, 3).tolist(), ML.flip_image(img_array, 'right').tolist())

        # two flips should recreate the original image
        self.assertEqual(img_array.tolist(), ML.flip_image(ML.flip_image(img_array, 'up'), 'up').tolist())
        self.assertEqual(img_array.tolist(), ML.flip_image(ML.flip_image(img_array, 'down'), 'down').tolist())
        self.assertEqual(img_array.tolist(), ML.flip_image(ML.flip_image(img_array, 'left'), 'left').tolist())
        self.assertEqual(img_array.tolist(), ML.flip_image(ML.flip_image(img_array, 'right'), 'right').tolist())

        # negative test
        with self.assertRaises(ValueError):
            ML.flip_image(img_array, 'asdf')
        with self.assertRaises(ValueError):
            ML.flip_image(img_array, 4)

    def test_remove_empty_columns(self):

        ML = MachineLearning
        array = np.zeros((100, 2))
        self.assertEqual(ML.remove_empty_columns(array)[0].shape, (100, 0))
        self.assertEqual(ML.remove_empty_columns(array)[1], [0, 1])

        array = np.ones((100, 2))
        self.assertEqual(ML.remove_empty_columns(array)[0].shape, (100, 2))
        self.assertEqual(ML.remove_empty_columns(array)[1], [])

        array = np.ones((100, 2))
        array[:, 0] = 0
        self.assertEqual(ML.remove_empty_columns(array)[0].shape, (100, 1))
        self.assertEqual(ML.remove_empty_columns(array)[1], [0])

        array = np.zeros((100, 4))

        row = np.zeros((100,))
        row[0:25] = 1
        array[:, 1] = row
        row[25:50] = 1
        array[:, 2] = row
        row[50:75] = 1
        array[:, 3] = row

        self.assertEqual(ML.remove_empty_columns(array, threshold=0.50)[0].shape, (100, 2))
        self.assertEqual(ML.remove_empty_columns(array, threshold=0.50)[1], [0, 1])
        self.assertEqual(ML.remove_empty_columns(array, threshold=0.51)[0].shape, (100, 1))
        self.assertEqual(ML.remove_empty_columns(array, threshold=0.51)[1], [0, 1, 2])

        array = np.zeros((100, ))
        with self.assertRaises(IndexError):
            ML.remove_empty_columns(array)
        array = np.zeros((1, 1, 1))
        with self.assertRaises(IndexError):
            ML.remove_empty_columns(array)

    def test_create_black_bar(self):
        ml = MachineLearning()

        attrs = ('black_bar1',
                 'black_bar3',
                 'black_bar4')

        for attr in attrs:
            self.assertIsNotNone(ml.__getattribute__(attr))
            ml.__setattr__(attr, None)

        for attr in attrs:
            self.assertIsNone(ml.__getattribute__(attr))

        ml._create_black_bar()
        for attr in attrs:
            self.assertIsNotNone(ml.__getattribute__(attr))




if __name__ == '__main__':
    unittest.main()
