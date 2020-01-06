import unittest
import warnings
import time
import socket
import os
from PIL import Image
import shutil
import pathlib
import multiprocessing
import threading

import NeuroMario
from MovieFile import MovieFile
from TimeFromImage import TimeFromImage

TMP_FOLDER = 'test/tmp/'


class NeuroMarioTestCases(unittest.TestCase):

    def setUp(self):
        full_dir = os.path.join(os.getcwd(), TMP_FOLDER)
        try:
            shutil.rmtree(full_dir)
        except:
            pass
        pathlib.Path(full_dir).mkdir(parents=True, exist_ok=True)
        cur_dir = os.path.join(os.getcwd(), 'emptyMovie')
        try:
            os.remove(os.path.isfile(os.path.join(cur_dir)))
            os.remove(os.path.isfile(os.path.join(cur_dir)))
        except:
            pass

    def tearDown(self):
        shutil.rmtree(os.path.join(os.getcwd(), TMP_FOLDER))

    def test_get_own_ip(self):
        self.assertIn('.', socket.gethostbyname(socket.gethostname()))

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

    def test_MovieFile(self):
        movie_file = MovieFile(filename='test/Super Mario Kart (USA).bk2')
        movie_file.parse_movie()
        self.assertEqual(len(movie_file.pressed_keys), movie_file.length)
        self.assertEqual(len(movie_file.pressed_keys), 3)
        self.assertEqual(movie_file.pressed_keys[0], b'|..|............|............|')
        self.assertEqual(movie_file.pressed_keys[1], b'|..|..L....B....|............|')
        self.assertEqual(movie_file.pressed_keys[2], b'|..|.......B....|............|')
        movie_file.pressed_keys = []
        self.assertEqual(movie_file.pressed_keys, [])
        movie_file.read_log_file('test/Input Log.txt')
        self.assertEqual(len(movie_file.pressed_keys), movie_file.length)
        self.assertEqual(len(movie_file.pressed_keys), 3)
        self.assertEqual(movie_file.pressed_keys[0], b'|..|............|............|')
        self.assertEqual(movie_file.pressed_keys[1], b'|..|..L....B....|............|')
        self.assertEqual(movie_file.pressed_keys[2], b'|..|.......B....|............|')

    def test_emuhawk(self):
        e = NeuroMario.Emuhawk()
        e.start()
        time.sleep(5)
        self.assertTrue(e.stop())

    def test_replay_socket(self):

        # test captured in test_TimeFromImage
        t = TimeFromImage()
        t.classify()

        # test captured in test_MovieFile
        movie_file = MovieFile(filename='test/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()
        
        g = NeuroMario.GameServer(socket_autostart=True,
                                  socket_ip=socket.gethostbyname(socket.gethostname()),
                                  socket_port=9980)
        e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                               socket_port=g.server.port,
                               lua_script=os.path.join(os.getcwd(), 'lua/replay_socket.lua'))
        e.start()
        g.server.create_connection()
        final_time = g.replay(movie_file.pressed_keys, method='socket')
        filename = 'test/tmp/tmp.png'
        with open(filename, 'wb') as f:
            f.write(final_time)
        self.assertIn('12437', t.predict_time_from_filenames([filename]))
        e.stop()

    @unittest.skip('temp')
    def test_replay_http(self):
        # test captured in test_TimeFromImage
        t = TimeFromImage()
        t.classify()

        # test captured in test_MovieFile
        movie_file = MovieFile(filename='test/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()

        e = NeuroMario.Emuhawk(http_ip=socket.gethostbyname(socket.gethostname()),
                               http_port=9990,
                               lua_script=os.path.join(os.getcwd(), 'listen_http.lua'))
        e.start()
        g = NeuroMario.GameServer(http_autostart=True,
                                  http_ip='',
                                  http_port=9990)

        final_time = g.replay(movie_file.pressed_keys, method='http')
        filename = os.path.join(TMP_FOLDER, 'tmp.png')
        with open(filename, 'wb') as f:
            f.write(final_time)
        self.assertIn('12437', t.predict_time_from_filenames([filename]))
        e.stop()

    @unittest.skip('temp')
    def test_defined_state(self):

        max_time = 60
        e = NeuroMario.Emuhawk()
        cur_dir = os.path.join(os.getcwd(), 'emptyMovie')
        self.assertFalse(os.path.isfile(os.path.join(cur_dir, 'unittest.bk2')))
        self.assertFalse(os.path.isfile(os.path.join(cur_dir, 'unittest.state')))
        e.create_defined_state(track='GHOST VALLEY 1', player='MARIO', filename_movie='unittest.bk2')
        self.assertTrue(e.is_running())
        while e.is_running() and max_time > 0:
            time.sleep(1)
            max_time -= 1
        self.assertGreater(max_time, 0)
        self.assertTrue(os.path.isfile(os.path.join(cur_dir, 'unittest.bk2')))
        self.assertTrue(os.path.isfile(os.path.join(cur_dir, 'unittest.state')))

    @unittest.skip('temp')
    def test_start_from_defined_state(self):
        max_time = 25
        filename_state = os.path.join(os.getcwd(), 'emptyMovie', 'unittest.state')
        e = NeuroMario.Emuhawk(socket_ip=socket.gethostbyname(socket.gethostname()),
                               socket_port=9990,
                               lua_script=os.path.join(os.getcwd(), 'short_run.lua'))
        e.create_defined_state(track='GHOST VALLEY 1', player='MARIO', filename_movie='unittest.bk2')
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

        filename = 'test/tmp_output/tmp.png'
        with open(filename, 'wb') as f:
            f.write(received)
        running_time = g.predict_running_time_from_file(filename)
        self.assertEqual(running_time, '000014')
        self.assertEqual(g.running_time_to_seconds(running_time), 0.14)

    @unittest.skip('not working, wrong state')
    def test_run_one_round(self):

        movie_file = MovieFile(filename='test/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()

        g = NeuroMario.GameServer(socket_autostart=True, socket_ip=socket.gethostbyname(socket.gethostname()),
                                  socket_port=9991)
        e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                               socket_port=g.server.port,
                               lua_script=os.path.join(os.getcwd(), 'lua/one_round_socket.lua'))
        filename_state = os.path.join(os.getcwd(), 'emptyMovie', 'unittest.state')
        e.state = filename_state
        e.start()
        g.server.create_connection()

        run_time = g.run_one_round('socket', 'replay', movie_file.pressed_keys[1047:])
        e.stop()
        self.assertEqual(run_time, 17.71)

    @unittest.skip('fails, TODO')
    def test_predict_finishline(self):

        test_cases = list()
        test_cases.append(['test/lap/super mario kart (usa).bk2_frame_3106.png', True])
        test_cases.append(['test/lap/super mario kart (usa).bk2_frame_4155.png', True])
        test_cases.append(['test/lap/super mario kart (usa).bk2_frame_5956.png', True])
        test_cases.append(['test/lap/super mario kart (usa)-2.bk2_frame_3467.png', True])
        test_cases.append(['test/lap/super mario kart (usa)-2.bk2_frame_6295.png', True])
        test_cases.append(['test/lap/super mario kart (usa)-8.bk2_frame_4650.png', True])
        test_cases.append(['test/lap/super mario kart (usa)-8.bk2_frame_2391.png', False])
        test_cases.append(['test/lap/super mario kart (usa)-8.bk2_frame_1996.png', False])
        test_cases.append(['test/lap/super mario kart (usa)-7.bk2_frame_5167.png', False])
        test_cases.append(['test/lap/super mario kart (usa).bk2_frame_2318.png', False])
        test_cases.append(['test/lap/super mario kart (usa).bk2_frame_3369.png', False])
        test_cases.append(['test/lap/super mario kart (usa)-2.bk2_frame_3601.png', False])

        g = NeuroMario.GameServer()
        for image, outcome in test_cases:
            p = g.predict_finishline_from_filename(image)
            self.assertEqual(p, outcome, 'failed prediction for image: {}'.format(image))

    def test_calculate_image_hash(self):
        g = NeuroMario.GameServer()
        hash1 = g.calculate_img_hash('test/rotate.png')
        hash2 = g.calculate_img_hash(Image.open('test/rotate.png'))
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash1.hash.shape[0], g.hashsize)
        self.assertEqual(hash1.hash.shape[1], g.hashsize)
        self.assertEqual(str(hash1), '3e1400287fffff00')

        g.hashsize = 4
        hash1 = g.calculate_img_hash('test/rotate.png')
        hash2 = g.calculate_img_hash(Image.open('test/rotate.png'))
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash1.hash.shape[0], g.hashsize)
        self.assertEqual(hash1.hash.shape[1], g.hashsize)

    @unittest.skip('not working, TODO')
    def test_calculate_hashes(self):
        g = NeuroMario.GameServer()
        self.assertIsNone(g.all_hashes)
        self.assertIsNone(g.hashes)
        self.assertIsNone(g.hash_to_file)
        f1 = os.path.join(TMP_FOLDER, 'hash.pickle')
        f2 = os.path.join(TMP_FOLDER, 'hash_to_file.pickle')
        g.calculate_hashes(filename_hashes=f1,
                           filename_hash_to_file=f2)
        self.assertTrue(os.path.isfile(f1))
        self.assertTrue(os.path.isfile(f2))

        # test pickling
        g2 = NeuroMario.GameServer()
        self.assertIsNone(g2.all_hashes)
        self.assertIsNone(g2.hashes)
        self.assertIsNone(g2.hash_to_file)
        g2.calculate_hashes(filename_hashes=f1,
                            filename_hash_to_file=f2)
        for i in range(0, len(g2.all_hashes), 10000):
            self.assertEqual(g2.all_hashes[i], g.all_hashes[i])

        # assert that we are actually working on two different dicts
        self.assertNotEqual(g2.all_hashes[0], g2.all_hashes[100])
        g2.all_hashes[0] = g2.all_hashes[100]
        self.assertNotEqual(g2.all_hashes[0], g.all_hashes[0])
        h = g.calculate_img_hash('test/lap/super mario kart (usa)-8.bk2_frame_4650.png')
        self.assertNotIn(h, g.all_hashes)
        self.assertNotIn(str(h), g.hashes)
        h = g.calculate_img_hash('test/lap/super mario kart (usa).bk2_frame_2318.png')
        self.assertIn(h, g.all_hashes)
        self.assertIn(str(h), g.hashes)

    def test_multi_threading(self):

        # start simple with two threads
        thread_number = 2
        game_servers = list()
        emuhawks = list()
        filename_state = os.path.join(os.getcwd(), 'states/GhostValley_2.State')
        movie_file = MovieFile(filename='test/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()
        threads = list()
        for i in range(thread_number):
            g = NeuroMario.GameServer(socket_autostart=True, socket_ip=socket.gethostbyname(socket.gethostname()),
                                      socket_port=9991 + i)
            e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                                   socket_port=g.server.port,
                                   lua_script=os.path.join(os.getcwd(), 'lua/one_round_socket.lua'))
            e.state = filename_state
            game_servers.append(g)
            emuhawks.append(e)
            e.start()
            g.server.create_connection()
            t = threading.Thread(target=g.run_one_round,
                                 args=('socket', 'replay', movie_file.pressed_keys[1047:])
                                 )

            threads.append(t)
            t.start()
        while len(threads) > 0:
            time.sleep(5)
            for i, tt in enumerate(threads):
                if not tt.is_alive():
                    threads.pop(i)
                    break

    def test_multi_processing(self):

        expected_values = {
            0: 18.40,
            1: 18.57,
            2: 18.60,
            3: 19.00
        }

        # use number of available cores
        cpus = multiprocessing.cpu_count()
        game_servers = list()
        emuhawks = list()
        filename_state = os.path.join(os.getcwd(), 'states/GhostValley_2.State')
        movie_file = MovieFile(filename='test/Super Mario Kart (USA)-3.bk2')
        movie_file.parse_movie()
        processes = list()
        return_values = multiprocessing.Array('f', cpus)
        for i in range(cpus):
            g = NeuroMario.GameServer(socket_autostart=True, socket_ip=socket.gethostbyname(socket.gethostname()),
                                      socket_port=9991 + i, verbose=False, printer='dummy')
            g.id = i
            e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                                   socket_port=g.server.port,
                                   lua_script=os.path.join(os.getcwd(), 'lua/one_round_socket.lua'))
            e.state = filename_state
            game_servers.append(g)
            emuhawks.append(e)
            e.start()
            g.server.create_connection()
            # slightly alter the input for each run
            for frame in range(0, (i % 4) * 16):
                movie_file.pressed_keys[2360 + frame] = movie_file.pressed_keys[1047]

            p = multiprocessing.Process(target=g.run_one_round,
                                        args=('socket', 'replay', movie_file.pressed_keys[1047:]),
                                        kwargs=dict(multiprocess=return_values))

            processes.append(p)
            p.start()

        for proc in processes:
            proc.join()
        for i in range(cpus):
            self.assertAlmostEqual(return_values[i], expected_values[i % 4], places=3,
                                   msg='value in pos {}, expected: {}, got: {}'.format(i, return_values[i],
                                                                                       expected_values[i % 4]))

    def test_neural_net(self):
        missing = [0, 1, 4, 5, 6, 8, 9, 11]
        g = NeuroMario.GameServer(socket_autostart=True,
                                  socket_ip=socket.gethostbyname(socket.gethostname()),
                                  socket_port=9990)
        e = NeuroMario.Emuhawk(socket_ip=g.server.ip,
                               socket_port=g.server.port,
                               lua_script=os.path.join(os.getcwd(), 'lua/listen_socket_screenshot.lua'),
                               )
        e.state = os.path.join(os.getcwd(), 'states/GhostValley_2.State')
        e.start()
        g.server.create_connection()
        self.assertEqual(g.ai(method='socket',
                              modelname='test/model_binary_crossentropy_keras.optimizers.Adam_sigmoid',
                              missing=missing,
                              threshold=0.15),
                         21.19)


if __name__ == '__main__':
    warnings.simplefilter("ignore", ResourceWarning)
    unittest.main()
