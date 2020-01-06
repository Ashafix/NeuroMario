import unittest
from TimeFromImage import TimeFromImage


class TestTimeFromImage(unittest.TestCase):
    def test_TimeFromImage(self):

        img = 'test/super mario kart (usa)-3.bk2_frame_6921.png'
        t = TimeFromImage()
        t.classify()
        self.assertTrue(t.test_classifier())
        pred1 = t.predict_time_from_filenames(img)
        self.assertEqual(len(pred1), 1)
        pred2 = t.predict_time_from_filenames([img])
        self.assertEqual(len(pred2), 1)
        self.assertDictEqual(pred1, pred2)
        self.assertEqual(list(pred1.keys())[0], '12437')
        self.assertEqual(list(pred1.values())[0], 1)
        pred3 = t.predict_time_from_filenames([img] * 3)
        self.assertEqual(len(pred3), 1)
        self.assertEqual(list(pred3.values())[0], 3)

        self.assertTrue(t.image_has_total(img))
        self.assertFalse(t.image_has_total('test/image1.png'))


if __name__ == '__main__':
    unittest.main()
