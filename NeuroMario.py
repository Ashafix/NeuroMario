import os
from PIL import Image
import numpy as np
import zipfile
import subprocess
import time
import stat
import random
import collections
import imagehash
import datetime
from scipy import ndimage, spatial
import io
import sys
import shutil
import pickle
import IPython.display
import os
import sklearn.svm
import sklearn.utils
import statistics
import ServerSocket
import socket
import signal
import functools
import ServerHTTP
import psutil
import io

class Printer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        pass

    def log(self, msg, verbose=None):
        if (verbose is True) or (self.verbose and verbose is not False):
            print(msg)


class Emuhawk:
    def __init__(self, emuhawk_exe=None,
                 base_dir='',
                 rom_name="SuperMario.sfc",
                 movies=None,
                 max_threads=100,
                 wait_time=5,
                 socket_ip=None,
                 socket_port=None,
                 http_ip=None,
                 http_port=None,
                 url_get=None,
                 url_post=None,
                 lua_script=None):

        if emuhawk_exe is not None:
            self.emuhawk_exe = emuhawk_exe
        else:
            emuhawk_exe = os.getenv('emuhawk_exe')
        if emuhawk_exe is not None:
            self.emuhawk_exe = emuhawk_exe
        else:
            emuhawk_exe = self.find_emuhawk_exe()
        if emuhawk_exe is None:
            raise ValueError
        else:
            self.emuhawk_exe = emuhawk_exe

        self.base_dir = base_dir
        self.params_png = ' --movie="{}" --dump-type=imagesequence --dump-name="{}" --dump-length={} --dump-close "{}"'
        self.movie = None
        self.movies = dict()
        self.state = None
        self.rom_name = rom_name
        if not os.path.isfile(self.rom_name):
            if os.path.isfile(os.path.join(os.path.dirname(self.emuhawk_exe), self.rom_name)):
                self.rom_name = os.path.join(os.path.dirname(self.emuhawk_exe), self.rom_name)
            else:
                raise ValueError('ROM could not be found')
        self.running_movies = list()
        self.max_threads = max_threads
        self.wait_time = wait_time
        self.printer = Printer(verbose=True)
        self.pid = None
        self.socket_ip = socket_ip
        self.socket_port = socket_port
        self.http_ip = http_ip
        self.http_port = http_port

        if http_ip is not None and http_port is not None:
            self.url_get = "http://{}:{}/get".format(http_ip, http_port)
            self.url_post = "http://{}:{}/post".format(http_ip, http_port)
        else:
            self.url_get = url_get
            self.url_post = url_post
        self.lua_script = lua_script
        if movies is not None:
            for movie in movies:
                self.append_movie(movie)

    def find_emuhawk_exe(self):
        possible_locations = ['../BizHawk/output/EmuHawk.exe']
        for possible_location in possible_locations:
            if possible_location.startswith('..'):
                possible_location = os.path.join(os.getcwd(), possible_location)
            if os.path.isfile(possible_location):
                return possible_location

    def movie_to_png(self, movie, output_filename):
        waiting_time = 0
        while len(self.running_movies) >= self.max_threads:
            if not self.remove_finished_jobs():
                time.sleep(self.wait_time)
            waiting_time += self.wait_time
            if waiting_time > 300:
                break

        cmd_call = self.emuhawk_exe + self.params_png.format(movie, output_filename, self.movies[movie].length,
                                                             self.rom_name)
        self.printer.log(cmd_call)
        #print(movie, os.path.join(os.path.split(output_filename)[0], os.path.split(movie)[1]))
        shutil.copy(movie, os.path.join(os.path.split(output_filename)[0], os.path.split(movie)[1]))
        p = subprocess.Popen(cmd_call)
        self.running_movies.append(dict(name=movie,
                                        filename=output_filename,
                                        length=self.movies[movie].length,
                                        process=p)

                                   )

    def movies_to_png(self, movies=list(), folders=list()):
        mismatch = 0
        if len(movies) == 0:
            movies = list(self.movies.keys())
        if len(folders) == 0:
            folders.append(os.path.join(self.base_dir, 'movies_images'))
            mismatch = 1
            for movie in movies:
                movie_name = movie.split('/')[-1]
                folders.append(os.path.join(self.base_dir, 'movies_images', movie_name))
        if len(folders) - len(movies) > mismatch:
            self.printer.log('movies and folders do not match')
            # TO DO, raise error
            return None

        for folder in folders:
            if not os.path.isdir(folder):
                try:
                    os.mkdir(folder)
                except:
                    pass

        for i, movie in enumerate(movies):
            movie_name = movie.split('/')[-1]
            output_filename = '{}/movies_images/{}/{}_frame.png'.format(self.base_dir, movie_name, movie_name)
            self.movie_to_png(movie, output_filename)

    def append_movie(self, movie):
        if movie in self.movies:
            return False
        self.movies[movie] = movie_file(filename=movie)
        return True

    def append_movies(self, movies):
        for movie in movies:
            self.append_movie(movie)

    def remove_finished_jobs(self):
        to_be_removed = list()
        for job_info in self.running_movies:
            filename = job_info['name'][0:job_info['name'].rfind('.')]  # prefix
            filename += str(job_info['length'])  # the length of the movie
            filename += job_info['name'][job_info['name'].rfind('.'):]  # suffix
            if os.path.isfile(filename):
                to_be_removed.append(self.running_movies.index(job_info[0]))
        for remove_me in to_be_removed:
            self.running_movies.pop(remove_me)
        return len(self.running_movies) == 0

    def wait_until_finished(self, timeout=300):
        while len(self.running_movies) > 0 and timeout > 0:
            self.remove_finished_jobs()
            if len(self.running_movies) > 0:
                time.sleep(self.wait_time)
                timeout += -self.wait_time
        return timeout >= 0

    def create_defined_state(self, gametype="1P", racetype="Time Trial", player="Mario", cc_class=None, cup="MUSHROOM CUP", track="GHOST VALLEY 1",
                             filename_movie=None, auto_start=True):
        """
        Creates and saves a defined state at the beginning of a race
        :return:
        """

        gametype = gametype.upper()
        racetype = racetype.upper()
        player = player.upper()
        cup = cup.upper()
        track = track.upper()

        if gametype not in ("1P", "2P"):
            raise ValueError
        if racetype not in ("TIME TRIAL", "MARIOKART GP"):
            raise ValueError
        if cc_class is not None and int(cc_class) not in (50, 100, 150):
            raise ValueError

        valid_players = ("MARIO", "PRINCESS", "BOWSER", "KOOPA", "LUIGI", "TOAD", "DONKEY", "NO IDEA")
        if player not in valid_players:
            raise ValueError
        valid_cups = ("MUSHROOM CUP", "FLOWER CUP", "STAR CUP")

        if cup not in valid_cups:
            raise ValueError
        valid_tracks = ["MARIO CIRCUIT 1", "DONUT PLAINS 1", "GHOST VALLEY 1", "BOWSER CASTLE 1", "MARIO CIRCUIT 2", "CHOCO ISLAND 1", "GHOST VALLEY 2", "DONUT PLAINS 2", "BOWSER CASTLE 2", "MARIO CIRCUIT 3", "KOOPA BEACH 1", "CHOCO ISLAND 2", "VANILLA LAKE 1", "BOWSER CASTLE 3", "MARIO CIRCUI 4"]
        if track is not None and track not in valid_tracks:
            raise ValueError

        if filename_movie is None:
            filename_movie = "DefinedState.bk2"
        if filename_movie.endswith(".bk2"):
            filename_state = filename_movie[0:-4] + ".state"
        else:
            filename_state = filename_movie + ".state"


        # create "Input log.txt" file
        with open("emptyMovie/Input Log.txt", 'r') as f:
            empty_log = f.readlines()
        output_lines = list()
        output_lines.append(empty_log[0].strip())
        output_lines.append(empty_log[1].strip())

        joypad_down    = "|..|.D..........|............|"
        joypad_left    = "|..|..L.........|............|"
        joypad_right   = "|..|...R........|............|"
        joypad_select  = "|..|.....S......|............|"
        joypad_default = "|..|............|............|"
        # select 1P or 2P
        output_lines.extend([joypad_default] * 262)
        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 30)
        if gametype == "2P":
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 30)

        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 30)

        # select Time Trial or Mariokart GP
        if racetype == "TIME TRIAL":
            output_lines.extend([joypad_down] * 8)
        output_lines.extend([joypad_default] * 60)
        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 60)

        # press YES
        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 120)

        # select player
        p = valid_players.index(player)
        if p > 3:
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 8)
        while p % 4 > 0:
            output_lines.extend([joypad_right] * 8)
            output_lines.extend([joypad_default] * 8)
            p -= 1

        # press OK
        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 31)
        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 240)
        c = valid_cups.index(cup)
        if c > 0:
            output_lines.extend([joypad_left] * 8)
            output_lines.extend([joypad_default] * 8)
        while c > 0:
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 8)

        output_lines.extend([joypad_right] * 8)
        output_lines.extend([joypad_default] * 30)

        # select track
        t = valid_tracks.index(track) % 5
        while t > 0:
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 8)
            t -= 1

        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 60)
        output_lines.extend([joypad_select] * 8)
        output_lines.extend([joypad_default] * 150)
        output_lines.append(empty_log[2].strip())

        with open(os.path.join(os.getcwd(), "emptyMovie", 'Input Log.txt_'), 'w') as f:
            f.write('\n'.join(output_lines))
        zf = zipfile.ZipFile(filename_movie, "w", zipfile.ZIP_DEFLATED)

        zf.write("SyncSettings.json", "SyncSettings.json")
        zf.write("Subtitles.txt", "Subtitles.txt")
        zf.write("Input Log.txt_", "Input Log.txt")
        zf.write("Comments.txt", "Comments.txt")
        zf.write("Header.txt", "Header.txt")
        zf.close()

        if auto_start:
            lua_string = """index = 0

while index < movie.length() do
	emu.frameadvance()
	index = index + 1
end

savestate.save("{}")
client.exit()
""".format(filename_state)
            filename_lua = os.path.join(os.getcwd(), "emptyMovie", 'defined_state.lua')
        with open(filename_lua, 'w') as f:
            f.write(lua_string)
        self.lua_script = filename_lua
        self.movie = filename_movie
        self.start()
        return filename_state

    def start(self):

        args = [self.emuhawk_exe]
        if self.socket_ip is not None or self.socket_port is not None:
            if self.socket_ip is None or self.socket_port is None:
               raise ValueError
            args.append('--socket_ip=')
            args[-1] += self.socket_ip
            args.append('--socket_port=')
            args[-1] += str(self.socket_port)

        if self.url_get is not None or self.url_post is not None:
            if self.url_get is None or self.url_post is None:
                raise ValueError
            args.append('--url_get=')
            args[-1] += self.url_get
            args.append('--url_post=')
            args[-1] += self.url_post
        if self.lua_script is not None:
            args.append('--lua=')
            args[-1] += self.lua_script
        if self.movie is not None:
            args.append('--movie=')
            args[-1] += self.movie
        if self.state is not None:
            args.append('--load-state=')
            args[-1] += self.state

        args.append(self.rom_name)
        print(args)
        p = subprocess.Popen(args)
        self.pid = p.pid

    def stop(self):
        if self.pid is None:
            return False

        try:
            subprocess.run("taskkill /F /T /PID {}".format(self.pid), shell=True)
            return True
        except:
            return False

    def is_running(self):
        return psutil.pid_exists(self.pid)

class GameServer:
    def __init__(self,
                 socket_autostart=False,
                 socket_port=9999,
                 socket_ip='',
                 http_autostart=False,
                 http_ip='',
                 http_port=9876,
                 directory=None,
                 verbose=True):
        """

        :param port:
        :param ip:
        """

        # use current directory as default
        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory

        self.hashes = None
        self.pressed_keys = dict()
        self.hash_to_file = None
        self.hashsize = 8
        self.image_method = 'hash'
        self.crop_box = (0, 0, 256, 112)
        self.new_index = 0
        self.advanced_listener_initialized = False
        self.finished = False
        if socket_autostart:
            self.server = ServerSocket.ServerSocket(ip=socket_ip,
                                                    port=socket_port)
        elif http_autostart:
            self.server = ServerHTTP.ServerHTTP(ip=http_ip,
                                                port=http_port)
        else:
            self.server = None
        self.connection = None
        self.address = None
        self.img = None
        self.image_obj = None
        self.finish_imgs = None
        self.adv_start_time = time.time()
        self.last_hash = None
        self.adv_fails = 0
        self.new_hash = ''
        self.advanced_listener_initialized = True
        self.hash_repeat = 0
        self.index = -1
        self.printer = Printer(verbose=verbose)
        self.cloud = False
        self.adv_start_time = time.time()
        self.last_hash = None
        self.adv_fails = 0
        self.new_hash = ''
        self.advanced_listener_initialized = False
        self.hash_repeat = 0
        self.index = -1
        self.cloud = False
        self.all_hashes = None
        self.valid_methods = ('socket', 'http', 'mmf')
        self.classifier= dict()
        self.hashes_finished = ['6838183c3e7fff00',
                                'ec251c1c3e7f7b00',
                                '703838383c7fff80',
                                '511858307cffff80',
                                'e4270c1c3e7f7f00',
                                '783838303cffff00',
                                '6c2d181c3e7f7f00',
                                '283c18383e7fff80',
                                'f4260e1c3e3f7f00',
                                '2c271c1c3e7f7f00',
                                'ec261c1c3e3f7f00',
                                '6839181c3e7fff00',
                                '703818307cffff80',
                                '683c18383e7fff00',
                                '6c38181c3e7fff00',
                                '383c38383c7fff00',
                                '2c2d1c1c3e7f7f00',
                                '2c3c181c3e7fff00',
                                'ec241c1c3e7f7f00',
                                'ec29181c3e7f7f00',
                                '6c271c1c3e7f7b00',
                                'ec261c1c3e7f7b00',
                                '6c261c1c3e7f7f00',
                                '6c2d0c1c3e7f7f00',
                                '711840707cffff80',
                                'ec28181c3e7fff00',
                                'f4260c1c3e7f7f00',
                                '783c38181c7fff80',
                                '6ca9181c3e7f7f00',
                                'ec270c1c3e7f7b00',
                                '6c251c1c3e7f7f00',
                                '2c39181c3e7fff00']
        if os.path.isdir(os.path.join(self.directory, 'movies_images', 'new')):
            pass
        with open('classifier_lap_rf.txt', 'rb') as f:
            self.classifier['lap'] = pickle.load(f)

        with open('classifier_runingtime.txt', 'rb') as f:
            self.classifier['runningtime'] = pickle.load(f)
        self.last_hash = ''
        self.new_hashes = list()

    @staticmethod
    def decoder_dummy(message):
        return message

    @staticmethod
    def decoder_post(message):
        return message.split('=')[-1]

    def replay(self, joypad_sequence, method=None,
               run_time=2*60):
        """
        Replays a sequence of joypad inputs
        :param joypad_sequence: a list of joypad inputs or a filename with a EmuHawk log file
        :param method: string, how to communicate with EmuHawk, either 'socket', 'http' or 'mmf', None means autodetect
        :return:
        """

        if method is None:
            if self.server is not None:
                if self.server.__class__ == ServerSocket.ServerSocket:
                    method = 'socket'

        if method is None or method not in self.valid_methods:
            raise ValueError('method needs to be one of: {}'.format(self.valid_methods))

        if method == 'socket':
            server_receive = self.server.receive
            decoder = self.decoder_dummy
            server_send = self.server.connection.send
            server_final = self.server.connection.close
        elif method == 'http':
            server_receive = self.server.receive
            decoder = self.decoder_post
            server_send = self.server.send
            server_final = str # just a dummy which does nothing

        start_time = time.time()
        index = 0
        while time.time() - start_time < run_time:
            buf = server_receive()
            if len(buf) > 0:
                try:
                    index = int(decoder(buf.decode()))
                    print(index)
                    if index < len(joypad_sequence):
                        resp = joypad_sequence[index]
                    else:
                        break

                except:
                    index = -1
                    resp = None

            if index > -1:
                not_send = 10
                while not_send > 0 and resp is not None:
                    try:
                        server_send(resp)
                        not_send = 0
                    except:
                        print('failed')
                        not_send -= 1

        server_send(b'screenshot')
        time.sleep(0.1)
        img_buf = server_receive(image=True)
        server_final()
        return img_buf

    def predict_finishline(self, image):
        """
        Predicts if the finish line was passed based on the appearance of the ghost
        :return:
        """

        return bool(self.classifier['lap'].predict(np.array(image.crop((0, 0, 250, 100))).reshape(1, -1)) == [1])

    def predict_finishline_from_filename(self, filename):
        """
        Predicts if the finish line was passed based on the appearance of the ghost
        :param filename: string with the image filename
        :return:
        """

        image = Image.open(filename).convert('L')
        return self.predict_finishline(image)

    def run_one_round(self, server, method, input_values):

        valid_servers = ("http", "socket", "mmf")
        if server.lower() not in valid_servers:
            raise ValueError("Server {} must be in {}".format(valid_servers))

        valid_methods = ("hash", "nn", "replay")
        if method.lower() not in valid_methods:
            raise ValueError("Method {} must be in {}".format(valid_methods))

        if server == 'socket':
            server_receive = self.server.receive
            decoder = self.decoder_dummy
            server_send = self.server.connection.send
            server_final = self.server.connection.close
        elif server == 'http':
            server_receive = self.server.receive
            decoder = self.decoder_post
            server_send = self.server.send
            server_final = str # just a dummy which does nothing

        if method == 'hash':
            # TO DO
            response_function = str
        elif method == 'nn':
            # TO DO
            response_function = str
        elif method == 'replay':
            decoder = int
            response_function = input_values.__getitem__


        start_time = time.time()

        finished = False
        index = 0
        run_time = 60
        while time.time() - start_time < run_time:
            buf = server_receive(image=True)
            if len(buf) > 0:
                with open('index_{}.png'.format(index), 'wb') as f:
                    f.write(buf)
                try:
                    Image.open(io.BytesIO(buf)).convert("L")
                except OSError as e:
                    print(e, index)

                    buf += server_receive()
                # check if round passed
                if self.predict_finishline(Image.open(io.BytesIO(buf)).convert("L")):
                    finished = True
                    break
                try:
                    #resp = response_function(decoder(buf.decode()))
                    resp = input_values[index]
                except:
                    break
                not_send = 10
                while not_send > 0 and resp is not None:
                    try:
                        server_send(resp)
                        not_send = 0
                    except:
                        print('failed')
                        not_send -= 1
                index += 1

        if finished:
            return self.running_time_to_seconds(self.predict_running_time(Image.open(io.BytesIO(buf)).convert("L")))

    def predict_running_time(self, image, output=False):
        crop_numbers = (176, 7, 242, 21)
        crop_digits = list()
        crop_digits.append(0)
        crop_digits.append(8)
        crop_digits.append(24)
        crop_digits.append(32)
        crop_digits.append(48)
        crop_digits.append(56)
        main_img = image.crop(crop_numbers)
        index = 0
        prediction = list()
        for i, x in enumerate(crop_digits):
            index += 1
            c = main_img.crop((x, 0, x + 8, 14))
            if output:
                IPython.display.display(c)
            prediction.append(self.classifier['runningtime'].predict(np.array(c).reshape((1, -1))))

        return ''.join([str(x[0]) for x in prediction])

    def predict_running_time_from_file(self, filename):
        """
        Predicts the current running time from a screenshot file
        :param filename: the filename with the image
        :return:
        """
        image = Image.open(filename).convert('L')
        return self.predict_running_time(image)

    @staticmethod
    def running_time_to_seconds(running_time):
        """

        :param running_time: list with len 6, digits as seen in time on screenshots
        :return: time in seconds
        """

        if len(running_time) != 6:
            return -1
        t = int(running_time[0]) * 10 * 60
        t += int(running_time[1]) * 60
        t += int(running_time[2]) * 10
        t += int(running_time[3])
        t += int(running_time[4]) / 10
        t += int(running_time[5]) / 100

        return t

    def advanced_listen(self, image_obj, run_time=600):

        if self.advanced_listener_initialized is not True:
            self.adv_start_time = time.time()
            self.last_hash = None
            self.adv_fails = 0
            self.new_hash = ''
            self.advanced_listener_initialized = True
            self.hash_repeat = 0
            self.index = -1
            self.cloud = False

        finish_line = False

        while (time.time() - self.adv_start_time) < run_time:
            self.index += 1

            self.image_obj = image_obj
            try:
                self.new_hash = str(self.calculate_img_hash(image_obj.crop((0, 25, 256, 224))))
            except:
                pass
            if self.new_hash == self.last_hash:
                self.hash_repeat += 1
            else:
                self.last_hash = self.new_hash
                self.hash_repeat = 0

            if self.index > 500:
                image_gray = image_obj.convert('L')
                if self.classifier['lap'].predict(np.array(image_gray.crop((0, 0, 250, 100))).reshape(1, -1)) == [1]:
                    if not self.cloud:
                        self.run_times.append(self.predict_running_time(image_gray))
                        self.hash_repeat = 10**6
                    self.cloud = True
                else:
                    if self.cloud:
                        self.printer.log('I stopped seeing clouds')
                    self.cloud = False
            img_hash = self.calculate_img_hash(image_obj)
            if str(img_hash) in self.hashes_finished:

                if not finish_line:
                    self.finish_imgs = list()
                    #t = self.time_from_image()

            #if finish_line:
                #if t.image_has_total(np.array(image_obj)):
                    #self.printer.log('total', verbose=verbose)
                    #self.finish_imgs.append(np.array(image_obj))
                 #   if len(self.finish_imgs) > 100:
                  #      self.printer.log(t.predict_time_from_filenames(self.finish_imgs), verbose)
                   #     self.advanced_listener_initialized = False
                    #    return b'finished'

            if self.hash_repeat == 10**6:
                self.printer.log('finished by seeing a cloud')
                self.advanced_listener_initialized = False
                resp = b'Restart'
            elif self.hash_repeat > 500:
                self.printer.log('failed due to repeats')
                self.advanced_listener_initialized = False
                resp = b'Restart'
            else:
                resp = self.hash_to_joypad(img_hash, learn=True, allow_random=False, deterministic=False)
            return resp

        self.printer.log('failed due to timeout')
        self.advanced_listener_initialized = False
        return b'Restart'

    def hash_to_joypad(self, image_hash, learn=True, allow_random=True, deterministic=False):
        """
        Converts a hash to joypad input
        :param hash: an image hash
        :return: a byte-string with joypad input
        """
        if self.hashes is None:
            self.calculate_hashes()

        possibilities = self.hashes.get(str(image_hash))
        if not possibilities or len(possibilities) == 0:
            min_hash = min(self.all_hashes, key=lambda x: abs(x - image_hash))

            # slower but deterministic
            if deterministic:
                possibilities = list()
                diff = np.sum(np.abs(image_hash - min_hash))
                for hash_ in self.all_hashes:
                    s = np.sum(np.abs(image_hash - hash_))
                    if s == diff:
                        possibilities.append(str(hash_))
                possibilities.sort()
                possibilities = self.hashes.get(possibilities[0])
            else:
                possibilities = self.hashes.get(str(min_hash))
            # stores the new hash but only if the system is supposed to remember
            if learn:
                self.hashes[str(image_hash)] = possibilities
                self.all_hashes.append(image_hash)
                self.new_hashes.append([image_hash, possibilities])
            self.printer.log('new hash')
        else:
            self.printer.log('hash found2')

        try:
            if learn:
                return random.choice(list(set(possibilities)))
            else:
                return possibilities[0]
        except:
            if allow_random:
                return bytes(self.default_joypad(), 'utf-8')
            else:
                return b'|..|............|............|'

    def img_to_joypad(self, image):
        """
        converts an image to a joypad input based on a specified method
        :param method:
        :return:
        """
        if self.image_method == 'hash':
            return self.img_hash(image)

        return self.default_joypad()

    def default_joypad(self, player=1):
        default = '|..|............|............|'
        random_input = '|..|UDLRs.YBXAlr|UDLRs.YBXAlr|'
        rand_range = ((player - 1) * 12 + 3 + player,
                      (player - 1) * 12 + 15 + player
                      )
        rand_replacement = random.randint(*rand_range)
        return default[0:rand_replacement] + random_input[rand_replacement] + default[rand_replacement + 1:]

    def calculate_img_hash(self, image):
        """

        :param image:
        :return:
        """
        if isinstance(image, str):
            image = Image.open(image)
        try:
            image = image.crop(self.crop_box)
        except:
            image = Image.new('RGB', (self.crop_box[2] - self.crop_box[0], self.crop_box[3] - self.crop_box[1]))
        image_hash = imagehash.whash(image, hash_size=self.hashsize)
        return image_hash

    def img_hash(self, image, learn=True):
        """
        predicts a joypad input based on a hashed image
        :param image:
        :return:
        """
        if self.hashes is None:
            self.calculate_hashes()
        start_time = time.time()
        image_hash = self.calculate_img_hash(image)

        possibilities = self.hashes.get(str(image_hash))
        if not possibilities or len(possibilities) == 0:
            new_hash = True
            min_hash = min(self.all_hashes, key=lambda x: abs(x - image_hash))

            possibilities = self.hashes.get(str(min_hash))
            # stores the new hash but only if the system is supposed to remember
            if learn:
                self.hashes[str(image_hash)] = possibilities
            self.all_hashes.append(image_hash)
            self.printer.log('new hash')
        else:
            new_hash = False
            self.printer.log('hash found3')
        if new_hash:
            self.new_index += 1
            filename = 'D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/new/new_{}.png'.format(self.new_index)
            while os.path.isfile(filename):
                self.new_index += 1
                filename = 'D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/new/new_{}.png'.format(self.new_index)

            image.save(filename)
            filename = 'D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/new/new_{}.hash'.format(self.new_index)
            with open(filename, 'wb') as f:
                pickle.dump(image_hash, f, protocol=pickle.HIGHEST_PROTOCOL)
            #filename will be used later
            filename = 'D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies_images/new/new_{}.txt'.format(self.new_index)

        if str(image_hash) in self.hashes_finished:
            self.printer.log('Reached finish line')
            self.finished = True

        choice = random.choice(list(set(possibilities)))
        self.printer.log(time.time() - start_time)
        try:
            self.hashes[str(image_hash)] = [choice]
            if new_hash:
                with open(filename, 'wb') as f:
                    f.write(choice)

            self.printer.log(choice)
            return choice
        except:
            self.printer.log('could not find input')
            return bytes(self.default_joypad(), 'utf-8')

    def calculate_hashes(self, overwrite=False):
        """

        :return:
        """
        self.hashes = collections.defaultdict(list)
        self.hash_to_file = collections.defaultdict(list)

        if os.path.isfile('c:/temp/hashes.pickle') and not overwrite:
            start_time = time.time()
            with open('c:/temp/hashes.pickle', 'rb') as f:
                self.hashes = pickle.load(f)
            with open('c:/temp/hash_to_file.pickle', 'rb') as f:
                self.hash_to_file = pickle.load(f)
            pickled = True

            self.printer.log('done pickling hashes')
            self.printer.log('Time to read pickle: {}'.format(time.time() - start_time))
        else:
            pickled = False

        for dir in os.listdir(os.path.join(self.directory, 'movies_images/')):
            if os.path.isdir(dir):
                start_time = time.time()
                if dir.endswith('.bk2'):
                    filename = os.path.join(self.directory, 'movies_images/', dir, dir)
                    if os.path.isfile(filename):
                        m = movie_file(filename=filename)
                        self.pressed_keys[filename.lower()] = self.filter_keys(m.pressed_keys)
                self.printer.log(time.time() - start_time)
        if not pickled and not overwrite:
            for root, dirs, files in os.walk(os.path.join(self.directory, 'movies_images/')):
                for file in files:
                    if file.endswith('.png') or file.endswith('.bmp') or file.endswith('.jpg'):
                        image = Image.open(os.path.join(root, file))
                        image = image.crop(self.crop_box)
                        movie_name = file[0:file.rfind('.bk2') + 4]
                        if movie_name != 'new':
                            index = int(file[file.rfind('_') + 1:file.rfind('.png')])
                            image_hash = str(imagehash.whash(image, hash_size=self.hashsize))
                            self.hashes[image_hash].append(self.pressed_keys[movie_name][index])
                            self.hash_to_file[image_hash].append(file)

        self.printer.log('Time for walking files: {}'.format(time.time() - start_time))
        start_time = time.time()

        self.all_hashes = [imagehash.hex_to_hash(i, hash_size=self.hashsize) for i in list(self.hashes.keys())]
        self.printer.log('Time for all hashes: {}'.format(time.time() - start_time))
        if not pickled or overwrite:
            with open('hashes.pickle', 'wb') as f:
                pickle.dump(self.hashes, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open('hash_to_file.pickle', 'wb') as f:
                pickle.dump(self.hash_to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    def filter_keys(self, pressed_keys, allowed=b'|..|UDLRs.YBXAlr|............|'):

        if isinstance(pressed_keys, str) or isinstance(pressed_keys, bytes):
            pressed_keys = [pressed_keys]
            string_input = True
        else:
            string_input = False

        for p, pressed in enumerate(pressed_keys):
            for i, button in enumerate(pressed):
                if button not in ('.', '|') and button != allowed[i:i + 1]:
                    # print(pressed[0:i], allowed[i], pressed[i + 1:])
                    pressed_keys[p] = pressed[0:i] + allowed[i:i + 1] + pressed[i + 1:]
        if string_input:
            return pressed_keys[0]
        else:
            return pressed_keys


def trim_image(images, player=1, prefix='trimmed_', overwrite=True, remove_timer=True, save_file=False):
    if isinstance(images, str):
        images = [images]

    for img in images:
        if not os.path.isfile(img):
            continue
        new_filename = os.path.join(os.path.dirname(img), prefix + os.path.split(img)[1])

        if not overwrite and save_file and os.path.isfile(new_filename):
            continue
        new_img = Image.open(img)

        crop_box = (0, int((player - 1) * (new_img.height / 2)), new_img.width, int(player * new_img.height / 2))

        cropped = new_img.crop(crop_box)
        if player == 1 and remove_timer:
            pix = np.array(cropped)
            pix[0:40, pix.shape[1] - 185:] = [0, 0, 0, 255]
            cropped = Image.fromarray(pix)
        if save_file:
            cropped.save(new_filename, new_filename.split('.')[-1])
        return cropped


class MovieFile:
    def __init__(self, filename=''):
        self.filename = filename
        self.pressed_keys = list()
        self.length = -1
        if filename:
            self.parse_movie()

    def parse_movie(self, log_filename='Input Log.txt'):
        zip_file = zipfile.ZipFile(self.filename)
        file_found = -1
        for file_index, filename in enumerate(zip_file.filelist):
            if filename.filename == 'Input Log.txt':
                file_found = file_index
        if file_found == -1:
            # to do, raise error
            return None
        self.read_log_file('Input Log.txt', file_zip=zip_file)

    def read_log_file(self, filename_log, file_zip=None):
        if file_zip is None:
            log_file = open(filename_log, 'rb')
        else:
            log_file = file_zip.open(filename_log)

        data = log_file.read().strip().splitlines()
        log_file.close()
        self.pressed_keys = data[2:-1]
        self.length = len(self.pressed_keys)
        return self.pressed_keys


def identical_images(image1, image2):
    img1 = Image.open(image1)
    img1.load()
    img2 = Image.open(image2)
    img2.load()
    data1 = np.asarray(img1)
    data2 = np.asarray(img2)
    return np.array_equal(data1, data2)


def remove_redundant_files(filenames, pressed_buttons):
    if len(filenames) != len(pressed_buttons):
        # TO DO, raise error
        return filenames, pressed_buttons
    redundant_filefound = True
    to_be_removed = list()
    while redundant_filefound:
        redundant_filefound = False
        for i in range(len(filenames) - 1):
            if pressed_buttons[i] == pressed_buttons[i + 1]:
                if identical_images(filenames[i], filenames[i + 1]):
                    to_be_removed.append(i)
                    redundant_filefound = True
        for i in to_be_removed:
            filenames.pop(i)
            pressed_buttons.pop(i)
    return filenames, pressed_buttons


class TimeFromImage:
    def __init__(self):

        self.classifier = None
        self.random_state = 42
        self.training_size = 0.8
        self.directories = list()
        self.digits = None
        self.digit_hashes = None
        self.crop_total = (79, 88, 104, 98)
        self.image_total = None

        self.index_ranges = list()
        self.numbers = list()
        self.index_ranges.append((6740, 10 ** 6))
        self.directories.append(
            'movies_images/Super Mario Kart (USA)-3.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01741')
        self.numbers[-1].append('01534')
        self.numbers[-1].append('01575')
        self.numbers[-1].append('02093')
        self.numbers[-1].append('01494')
        self.numbers[-1].append('12437')

        self.index_ranges.append((6559, 10 ** 6))
        self.directories.append(
            'movies_images/Super Mario Kart (USA)-5.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01809')
        self.numbers[-1].append('01660')
        self.numbers[-1].append('01547')
        self.numbers[-1].append('01464')
        self.numbers[-1].append('01587')
        self.numbers[-1].append('12067')

        self.index_ranges.append((5601, 10 ** 6))
        self.directories.append(
            'movies_images/Super Mario Kart (USA)-6.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01653')
        self.numbers[-1].append('01477')
        self.numbers[-1].append('01654')
        self.numbers[-1].append('01522')
        self.numbers[-1].append('01476')
        self.numbers[-1].append('11782')

        self.index_ranges.append((5788, 10 ** 6))
        self.directories.append(
            'movies_images/Super Mario Kart (USA)-7.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01640')
        self.numbers[-1].append('01480')
        self.numbers[-1].append('02072')
        self.numbers[-1].append('01467')
        self.numbers[-1].append('01465')
        self.numbers[-1].append('12124')

        self.index_ranges.append((5746, 10 ** 6))
        self.directories.append(
            'movies_images/Super Mario Kart (USA)-8.bk2/')
        self.numbers.append(list())
        self.numbers[-1].append('01661')
        self.numbers[-1].append('01634')
        self.numbers[-1].append('01455')
        self.numbers[-1].append('01758')
        self.numbers[-1].append('01488')
        self.numbers[-1].append('11996')

        self.index_ranges.append((0, 10 ** 6))
        self.directories.append(
            'number_training/1/')
        self.numbers.append(list())
        self.numbers[-1].append('11030')
        self.numbers[-1].append('10142')
        self.numbers[-1].append('10205')
        self.numbers[-1].append('05693')
        self.numbers[-1].append('10723')
        self.numbers[-1].append('51793')

        self.index_ranges.append((0, 10 ** 6))
        self.directories.append(
            'number_training/2/')
        self.numbers.append(list())
        self.numbers[-1].append('11030')
        self.numbers[-1].append('10142')
        self.numbers[-1].append('10205')
        self.numbers[-1].append('05693')
        self.numbers[-1].append('11456')
        self.numbers[-1].append('52526')

        self.filenames = list()

        self.dig_pos = list()
        for i in range(5):
            self.dig_pos.append((0, i * 9, 8, i * 9 + 8))
            self.dig_pos.append((16, i * 9, 24, i * 9 + 8))
            self.dig_pos.append((24, i * 9, 32, i * 9 + 8))
            self.dig_pos.append((40, i * 9, 48, i * 9 + 8))
            self.dig_pos.append((48, i * 9, 56, i * 9 + 8))

        i = 52
        self.dig_pos.append((0, i, 8, i + 8))
        self.dig_pos.append((16, i, 24, i + 8))
        self.dig_pos.append((24, i, 32, i + 8))
        self.dig_pos.append((40, i, 48, i + 8))
        self.dig_pos.append((48, i, 56, i + 8))

        for movie, direc in enumerate(self.directories):
            self.filenames.append(list())

            for a, b, c in os.walk(direc):
                for filename in c:
                    if filename.endswith('.png'):
                        if '_frame_' in filename:
                            index = int(filename.split('_frame_')[1].split('.')[0])
                        else:
                            index = int(filename.split('_')[-1].split('.')[0])
                        if index >= self.index_ranges[movie][0] and index < self.index_ranges[movie][1]:
                            self.filenames[-1].append(os.path.join(direc, filename))

    def image_arrays_from_file(self, filename, movie, numbers):
        x = 112
        y = 37
        width = 60
        height = 60
        if isinstance(filename, str):
            img = Image.open(filename).crop((x, y, x + width, y + height)).convert('LA')
        elif isinstance(filename, np.ndarray):
            img = Image.fromarray(filename).crop((x, y, x + width, y + height)).convert('LA')
        else:
            raise ValueError('filename must be string or numpy array')
        digits = list()
        digit_hashes = list()
        sorted_images = list()
        for _ in range(10):
            digits.append(list())
            digit_hashes.append(list())

        for i, pos in enumerate(self.dig_pos):
            digit_array = np.array(img.crop(pos))[:, :, 0]

            h = hash(digit_array.tostring())
            n = int(numbers[movie][i // 5][i % 5])
            if h not in self.digit_hashes[n]:
                self.digits[n].append(digit_array)
                self.digit_hashes[n].append(h)

            digits[n].append(digit_array)
            digit_hashes[n].append(h)
            sorted_images.append(digit_array)

        return digits, digit_hashes, sorted_images

    def cut_image_arrays_from_file(self, filename):
        x = 112
        y = 37
        width = 60
        height = 60
        if isinstance(filename, str):
            img = Image.open(filename).crop((x, y, x + width, y + height)).convert('LA')
        elif isinstance(filename, np.ndarray):
            img = Image.fromarray(filename).crop((x, y, x + width, y + height)).convert('LA')

        sorted_images = list()

        for pos in self.dig_pos:
            digit_array = np.array(img.crop(pos))[:, :, 0]
            sorted_images.append(digit_array)

        return sorted_images

    def filenames_to_digits(self, filenames, numbers):

        self.digits = list()
        self.digit_hashes = list()

        for _ in range(10):
            self.digits.append(list())
            self.digit_hashes.append(list())

        a = False
        for movie, filename in enumerate(filenames):
            for file in filename:
                self.image_arrays_from_file(file, movie, numbers)
                # return self.digits

    def predict_time_from_filenames(self, filenames):
        if self.classifier is None:
            self.classify()
        if not isinstance(filenames, list):
            filenames = [filenames]
        solution = collections.defaultdict(int)
        for filename in filenames:
            digit_arrays = self.cut_image_arrays_from_file(filename)[-5:]
            solution[
                ''.join([str(s)[0] for s in list(self.classifier.predict(np.array(digit_arrays).reshape(5, -1)))])] += 1

        return solution

    @staticmethod
    def timestring_to_time(timestring):

        t = 0
        t += int(timestring[0]) * 60
        t += int(timestring[1]) * 10
        t += int(timestring[2])
        t += int(timestring[3]) * 0.1
        t += int(timestring[4]) * 0.01
        return t

    def test_classifier(self):
        for i, filename in enumerate(self.filenames):
            prediction = self.predict_time_from_filenames(filename)
            if len(prediction) == 0:
                continue
            best_score = max([int(i) for i in list(prediction)])
            if best_score != int(self.numbers[i][5]):
                print('failed')
                print('Expected: {}\nGot:{}'.format(self.numbers[i][5], prediction))
                print(i)
                return False
        return True

    def classify(self):

        self.filenames_to_digits(self.filenames, self.numbers)
        self.classifier = sklearn.svm.SVC(gamma=0.001)
        self.total_samples = 0
        for i in range(10):
            self.total_samples += len(self.digits[i])

        self.all_images = np.zeros([self.total_samples, 8, 8])
        self.all_digits = np.zeros([self.total_samples])
        index = 0
        for i in range(10):
            for d in self.digits[i]:
                self.all_images[index] = d
                self.all_digits[index] = i
                index += 1
        self.data_s, self.digits_s = sklearn.utils.shuffle(self.all_images.reshape(self.total_samples, -1),
                                                           self.all_digits, random_state=self.random_state)
        self.classifier.fit(self.data_s[0:int(self.total_samples * self.training_size)],
                            np.ravel(self.digits_s[0:int(self.total_samples * self.training_size)]))

    def get_image_total(self,
                        movie='Super Mario Kart (USA)-3.bk2',
                        index='6938',
                        movie_dir='movies_images/'):

        filename = '{0}{1}/{1}_frame_{2}.png'.format(movie_dir, movie, index)
        img = Image.open(filename).crop(self.crop_total)
        self.image_total = np.array(img)[:, :, :3]

    def image_has_total(self, img):
        """
        Checks whether an image has the total running time
        :param img: either an numpy array, Image object or string with filename
        :return: boolean, True if total running time is displayed in image, false if not
        """
        if isinstance(img, str):
            img = np.array(Image.open(img))[:, :, :3]
        if not isinstance(img, np.ndarray):
            img = np.array(img)[:, :, :3]
        if self.image_total is None:
            self.get_image_total()

        if np.shape(img) != (10, 25, 3):
            try:
                img = np.array(Image.fromarray(img).crop(self.crop_total))
            except:
                return False
        return np.sum(np.equal(img, self.image_total)) >= 453


if __name__ == '__main__':

    e = Emuhawk()
    e.create_defined_state(track="GHOST VALLEY 1", player="MARIO")
    sys.exit()
    index = 0
    while True:
        g = GameServer()
        g.calculate_hashes()
        print('finished hashing')
        g.create_connection()
        print('created connection, starting to listen')
        g.listen()
        with open('hashes_{}.pickle'.format(index), 'wb') as f:
            pickle.dump(g.hashes, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('finished_{}.pickle'.format(index), 'w') as f:
            f.write(str(g.finished))
        index += 1
    #e = emuhawk()
    #e.emuhawk_exe = 'C:/Users/ashafix/Downloads/BizHawk-1.13.0/EmuHawk.exe'
    #e.base_dir = 'D:/Users/Ashafix/Documents/GitHub/NeuroMario'
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA).bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-2.bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-3.bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-4.bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-5.bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-6.bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-7.bk2')
    #e.append_movie('D:/Users/Ashafix/Documents/GitHub/NeuroMario/movies/Super Mario Kart (USA)-8.bk2')
    #e.movies_to_png()

