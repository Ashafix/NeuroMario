import unittest
from utils import identical_images


class TestUtils(unittest.TestCase):
    def test_identical_image(self):
        img1 = 'test/image1.png'
        img2 = 'test/image2.png'
        img3 = 'test/image3.png'
        self.assertTrue(identical_images(img1, img2))
        self.assertTrue(identical_images(img1, img1))
        self.assertTrue(identical_images(img2, img2))
        self.assertTrue(identical_images(img3, img3))
        self.assertFalse(identical_images(img1, img3))
        self.assertFalse(identical_images(img2, img3))


if __name__ == '__main__':
    unittest.main()
