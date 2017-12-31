import unittest
import NeuroMario
import time
import socket
import os

class NeuroMarioTestCases(unittest.TestCase):

    @unittest.skip('working')   
    def test_get_own_ip(self):
        self.assertIn('.', socket.gethostbyname(socket.gethostname()))

    @unittest.skip('working')   
    def test_running_time_to_seconds(self):
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds(''), -1)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([]), -1)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([0, 0, 0, 0, 0, 0]), 0)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([0, 0, 0, 0, 0, 1]), 1 / 100)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([0, 0, 0, 0, 1, 0]), 1 / 10)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([0, 0, 0, 1, 0, 0]), 1)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([0, 0, 1, 0, 0, 0]), 10)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([0, 1, 0, 0, 0, 0]), 60)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 0, 0, 0, 0, 0]), 600)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 1, 0, 0, 0, 0]), 600 + 60)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 1, 1, 0, 0, 0]), 600 + 60 + 10)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 1, 1, 1, 0, 0]), 600 + 60 + 10 + 1)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 1, 1, 1, 1, 0]), 600 + 60 + 10 + 1 + 1/10)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 1, 1, 1, 1, 1]), 600 + 60 + 10 + 1 + 1/10 + 1/100)
        self.assertEqual(NeuroMario.GameServer.running_time_to_seconds([1, 1, 5, 9, 9, 9]), 600 + 60 + 5 * 10 + 9 * 1 + 9/10 + 9/100)

    @unittest.skip('working')
    def test_MovieFile(self):
        movie_file = NeuroMario.MovieFile(filename='tests/Super Mario Kart (USA).bk2')
        movie_file.parse_movie()
        self.assertEqual(len(movie_file.pressed_keys), movie_file.length)
        self.assertEqual(len(movie_file.pressed_keys), 3)
        self.assertEqual(movie_file.pressed_keys[0], b'|..|............|............|')
        self.assertEqual(movie_file.pressed_keys[1], b'|..|..L....B....|............|')
        self.assertEqual(movie_file.pressed_keys[2], b'|..|.......B....|............|')
        movie_file.pressed_keys = []
        self.assertEqual(movie_file.pressed_keys, [])
        movie_file.read_log_file('tests/Input Log.txt')
        self.assertEqual(len(movie_file.pressed_keys), movie_file.length)
        self.assertEqual(len(movie_file.pressed_keys), 3)
        self.assertEqual(movie_file.pressed_keys[0], b'|..|............|............|')
        self.assertEqual(movie_file.pressed_keys[1], b'|..|..L....B....|............|')
        self.assertEqual(movie_file.pressed_keys[2], b'|..|.......B....|............|')

    @unittest.skip('working')
    def test_identical_image(self):
        img1 = 'tests/image1.png'
        img2 = 'tests/image2.png'
        img3 = 'tests/image3.png'
        self.assertTrue(NeuroMario.identical_images(img1, img2))
        self.assertTrue(NeuroMario.identical_images(img1, img1))
        self.assertTrue(NeuroMario.identical_images(img2, img2))
        self.assertTrue(NeuroMario.identical_images(img3, img3))
        self.assertFalse(NeuroMario.identical_images(img1, img3))
        self.assertFalse(NeuroMario.identical_images(img2, img3))

    @unittest.skip('working')
    def test_TimeFromImage(self):

        img = 'tests/super mario kart (usa)-3.bk2_frame_6921.png'
        t = NeuroMario.TimeFromImage()
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
        self.assertFalse(t.image_has_total('tests/image1.png'))

    @unittest.skip('working')
    def test_emuhawk(self):
        e = NeuroMario.Emuhawk()
        e.start()
        time.sleep(5)
        self.assertTrue(e.stop())

    @unittest.skip('working')
    def test_replay_socket(self):

        # test captured in test_TimeFromImage
        t = NeuroMario.TimeFromImage()
        t.classify()

        # test captured in test_MovieFile
        movie_file = NeuroMario.MovieFile(filename='tests/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()
        
        g = NeuroMario.GameServer(socket_autostart=True, socket_ip=socket.gethostbyname(socket.gethostname()),
                                  socket_port=9990)
        e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                               socket_port=g.server.port,
                               lua_script=os.path.join(os.getcwd(), 'listen.lua'))
        e.start()
        g.server.create_connection()
        final_time = g.replay(movie_file.pressed_keys, method='socket')
        filename = 'tests/tmp_output/tmp.png'
        with open(filename, 'wb') as f:
            f.write(final_time)
        self.assertIn('12437', t.predict_time_from_filenames([filename]))
        e.stop()

    @unittest.skip('not working')
    def test_replay_http(self):
        # test captured in test_TimeFromImage
        t = NeuroMario.TimeFromImage()
        t.classify()

        # test captured in test_MovieFile
        movie_file = NeuroMario.MovieFile(filename='tests/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()

        e = NeuroMario.Emuhawk(http_ip=socket.gethostbyname(socket.gethostname()),
                               http_port=9990,
                               lua_script=os.path.join(os.getcwd(), 'listen_http.lua'))
        e.start()
        g = NeuroMario.GameServer(http_autostart=True,
                                  http_ip="",
                                  http_port=9990)

        final_time = g.replay(movie_file.pressed_keys, method='http')
        filename = 'tests/tmp_output/tmp.png'
        with open(filename, 'wb') as f:
            f.write(final_time)
        self.assertIn('12437', t.predict_time_from_filenames([filename]))
        e.stop()

    @unittest.skip('working')
    def test_defined_state(self):

        max_time = 60
        e = NeuroMario.Emuhawk()
        cur_dir = os.path.join(os.getcwd(), "emptyMovie")
        self.assertFalse(os.path.isfile(os.path.join(cur_dir, "unittest.bk2")))
        self.assertFalse(os.path.isfile(os.path.join(cur_dir, "unittest.state")))
        e.create_defined_state(track="GHOST VALLEY 1", player="MARIO", filename_movie="unittest.bk2")
        self.assertTrue(e.is_running())
        while e.is_running() and max_time > 0:
            time.sleep(1)
            max_time -= 1
        self.assertGreater(max_time, 0)
        self.assertTrue(os.path.isfile(os.path.join(cur_dir, "unittest.bk2")))
        self.assertTrue(os.path.isfile(os.path.join(cur_dir, "unittest.state")))

    @unittest.skip('working')
    def test_start_from_defined_state(self):
        max_time = 25
        filename_state = os.path.join(os.getcwd(), "emptyMovie", "unittest.state")
        e = NeuroMario.Emuhawk(socket_ip=socket.gethostbyname(socket.gethostname()),
                               socket_port=9990,
                               lua_script=os.path.join(os.getcwd(), 'short_run.lua'))
        g = NeuroMario.GameServer(socket_autostart=True, socket_ip=socket.gethostbyname(socket.gethostname()),
                                  socket_port=9990)
        e.state = filename_state
        e.start()
        g.server.create_connection()
        received = b''
        while max_time > 0 and received == b'':
            received = g.server.receive(image=True)
            time.sleep(0.01)
            max_time -= 0.01
        e.stop()

        self.assertGreater(max_time, 0)
        self.assertNotEqual(received, b'')

        filename = 'tests/tmp_output/tmp.png'
        with open(filename, 'wb') as f:
            f.write(received)
        running_time = g.predict_running_time_from_file(filename)
        self.assertEqual(running_time, "000014")
        self.assertEqual(g.running_time_to_seconds(running_time), 0.14)

    #@unittest.skip('not working')
    def test_run_one_round(self):

        movie_file = NeuroMario.MovieFile(filename='tests/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()

        g = NeuroMario.GameServer(socket_autostart=True, socket_ip=socket.gethostbyname(socket.gethostname()),
                                  socket_port=9990)
        e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                               socket_port=g.server.port,
                               lua_script=os.path.join(os.getcwd(), 'one_round_socket.lua'))
        filename_state = os.path.join(os.getcwd(), "emptyMovie", "unittest.state")
        e.state = filename_state
        e.start()
        g.server.create_connection()

        run_time = g.run_one_round("socket", "replay", movie_file.pressed_keys[1047:])
        e.stop()
        self.assertEqual(run_time, 17.71)

    def test_predict_finishline(self):

        test_cases = list()
        test_cases.append(["tests/lap/super mario kart (usa).bk2_frame_3106.png", True])
        test_cases.append(["tests/lap/super mario kart (usa).bk2_frame_4155.png", True])
        test_cases.append(["tests/lap/super mario kart (usa).bk2_frame_5956.png", True])
        test_cases.append(["tests/lap/super mario kart (usa)-2.bk2_frame_3467.png", True])
        test_cases.append(["tests/lap/super mario kart (usa)-2.bk2_frame_6295.png", True])
        test_cases.append(["tests/lap/super mario kart (usa)-8.bk2_frame_4650.png", True])
        test_cases.append(["tests/lap/super mario kart (usa)-8.bk2_frame_2391.png", False])
        test_cases.append(["tests/lap/super mario kart (usa)-8.bk2_frame_1996.png", False])
        test_cases.append(["tests/lap/super mario kart (usa)-7.bk2_frame_5167.png", False])
        test_cases.append(["tests/lap/super mario kart (usa).bk2_frame_2318.png", False])
        test_cases.append(["tests/lap/super mario kart (usa).bk2_frame_3369.png", False])
        test_cases.append(["tests/lap/super mario kart (usa)-2.bk2_frame_3601.png", False])

        g = NeuroMario.GameServer()
        for t in test_cases:
            p = g.predict_finishline_from_filename(t[0])
            self.assertEqual(p, t[1])

if __name__ == '__main__':
    unittest.main()
    
