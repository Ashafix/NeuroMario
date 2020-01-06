import os
import io
import time
import random
import pickle
import pathlib
import collections
import numpy as np
import imagehash
from PIL import Image
import yaml
import warnings

from ServerSocket import ServerSocket
from ServerHTTP import ServerHTTP
from Printer import Printer, PrinterDummy
from Joypad import Joypad
from MachineLearning import MachineLearning
from MovieFile import MovieFile

try:
    import IPython.display
except ImportError:
    warnings.warn('Interactive IPython mode disabled')


class GameServer:
    def __init__(self,
                 socket_autostart=False,
                 socket_port=9999,
                 socket_ip='',
                 http_autostart=False,
                 http_ip='',
                 http_port=9876,
                 directory=None,
                 verbose=True,
                 printer=None):
        """

        :param http_port:
        :param http_ip:
        """

        # use current directory as default
        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory
        self.init_directories()
        self.hashes = None
        self.pressed_keys = {}
        self.hash_to_file = None
        self.hashsize = 8
        self.image_method = 'hash'
        self.crop_box = (0, 0, 256, 112)
        self.new_index = 0
        self.advanced_listener_initialized = False
        self.finished = False
        if printer is None:
            self.printer = Printer(verbose=verbose)
        elif printer == 'dummy':
            self.printer = PrinterDummy()
        else:
            self.printer = printer
        if not verbose:
            printer_ = 'dummy'
        else:
            printer_ = None
        if socket_autostart:
            self.server = ServerSocket(ip=socket_ip,
                                       port=socket_port,
                                       printer=printer_)
        elif http_autostart:
            self.server = ServerHTTP(ip=http_ip,
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
        self.advanced_listener_initialized = True
        self.hash_repeat = 0
        self.index = -1

        self.cloud = False
        self.adv_start_time = time.time()
        self.new_hash = ''
        self.advanced_listener_initialized = False
        self.all_hashes = None
        self.valid_methods = ('socket', 'http', 'mmf')
        self.classifier = {}
        self.hashes_finished = None
        with open('classifiers/new_lap_rfc.pickle', 'rb') as f:
            self.classifier['lap'] = pickle.load(f)
            self.classifier['lap'].verbose = False
        with open('classifiers/new_lap_extended_rfc.pickle', 'rb') as f:
            self.classifier['lap2'] = pickle.load(f)
            self.classifier['lap2'].verbose = False
        with open('classifier_runingtime.txt', 'rb') as f:
            self.classifier['runningtime'] = pickle.load(f)
        self.last_hash = ''
        self.new_hashes = []
        self.false_positives = None
        self.true_positives = None
        # used for multiprcoessing identification
        self.id = None
        self.run_times = []
        self.__read_config()

    def __str__(self):
        resp = '{}: Server: {}'.format(type(self).__name__,
                                       self.server)
        return resp

    def __repr__(self):
        return self.__str__()

    def __read_config(self, filename='NeuroMarioConfig.yaml'):
        """
        Reads the yaml config file and adds attributes
        :return: None
        """
        with open(filename, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        self.false_positives = yaml_dict.get('finish_line_false_positives')
        self.true_positives = yaml_dict.get('finish_line_true_positives')
        self.hashes_finished = yaml_dict.get('hashes_finished')

    def init_directories(self):
        directories = ('movies_images', 'movies_images\\new', 'hashes', 'movies')
        for direc in directories:
            full_direc = os.path.join(self.directory, direc)
            pathlib.Path(full_direc).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def decoder_dummy(message):
        return message

    @staticmethod
    def decoder_post(message):
        return message.split('=')[-1]

    def ai(self, method=None, run_time=5*60, max_index=5000, modelname='model', model=None,
           threshold=0.4, missing=None, bit_array=False, rounds=1):
        """

        :param method:
        :param run_time: int, max time in seconds for which a simulation is performed,
                         afterwards the simulation is terminated
        :param max_index: int, maximum number of frames to use, afterwards the simulation is terminated
        :param modelname: string
        :param model:
        :param threshold: float
        :param missing:
        :param bit_array: boolean, whether the output array is a bit array, see MachineLearning.input_output
        :param rounds: int, number of rounds to run
        :return:
        """
        if method == 'socket':
            server_receive = self.server.receive
            decoder = self.decoder_dummy
            server_send = self.server.connection.send
        elif method == 'http':
            server_receive = self.server.receive
            decoder = self.decoder_post
            server_send = self.server.send
        else:
            raise ValueError('method needs to be one of: {}'.format(self.valid_methods))

        if model is None:
            import keras.models
            with open('{}.json'.format(modelname), 'r') as f:
                model = keras.models.model_from_json(f.read())

            model.load_weights('{}_weights.h5'.format(modelname))

        start_time = time.time()
        index = 0

        ml = MachineLearning()
        index_finishline_passed = 0
        while time.time() - start_time < run_time:

            buf = server_receive(image=True)
            img = None
            try:
                img = Image.open(io.BytesIO(buf))
                img = img.convert('L')
                index += 1

            except OSError as e:
                self.printer.log((e, index))
                buf += server_receive()

            # check if finish line was passed
            if img is not None and index > (index_finishline_passed + 1000) and self.predict_finishline(img):
                index_finishline_passed = index
                rounds -= 1
                if rounds <= 0:
                    self.printer.log('breaking because finish line was recognized')
                    break
            try:
                resp = model.predict(ml.prepare_image(img, normalize=True, gray=False).reshape(1, 112, 256, 1))[0]
            except Exception as e:
                self.printer.log(e)
                return 100
            if index > max_index:
                return 100
            if resp is not None:
                not_send = 10
                while not_send > 0:
                    joypad_output = Joypad.array_to_joypad(resp, threshold=threshold,
                                                           missing=missing, bit_array=bit_array)
                    if joypad_output == Joypad.empty:
                        joypad_output = Joypad.B
                    try:
                        server_send(joypad_output)
                        not_send = 0
                    except:
                        self.printer.log('failed')
                        not_send -= 1
        return self.running_time_to_seconds(self.predict_running_time(img.convert('L')))

    def replay(self, joypad_sequence, method=None,
               run_time=2*60):
        """
        Replays a sequence of joypad inputs
        :param joypad_sequence: a list of joypad inputs or a filename with a EmuHawk log file
        :param method: string, how to communicate with EmuHawk, either 'socket', 'http' or 'mmf', None means autodetect
        :param run_time: int, run time in seconds
        :return: , last image buffer
        """

        if method is None:
            if self.server is not None:
                if self.server.__class__ == ServerSocket:
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
        else:
            raise ValueError('method must be socket or http')

        start_time = time.time()
        index = 0
        resp = None
        failed = False

        while time.time() - start_time < run_time:
            buf = server_receive(packet_size=1024)
            if len(buf) > 0:
                try:
                    index = int(decoder(buf.decode()))
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
                        failed = False
                    except:
                        self.printer.log('failed')
                        not_send -= 1
                        failed = (not_send == 0)
                if failed:
                    self.printer.log('giving up')
                    break

        server_send(b'screenshot')
        time.sleep(0.1)
        img_buf = server_receive(image=True)
        try:
            server_final()
        except: # that's fine
            pass
        return img_buf

    def predict_finishline(self, image):
        """
        Predicts if the finish line was passed based on the appearance of the ghost
        :param image: PIL Image
        :return: bool, True if finish line was passed, False if not
        """

        img_cropped = image.crop((0, 0, 250, 100))
        img_array = np.array(img_cropped).reshape(1, -1)
        finish_line = bool(self.classifier['lap'].predict(img_array) == [1])
        if finish_line:
            hash_value = str(imagehash.phash(img_cropped, hash_size=5))
            self.printer.log(hash_value)

            if hash_value in self.true_positives and hash_value in self.false_positives:
                raise ValueError("hash {} was found in both false and true positives".format(hash_value))

            direc = os.path.join(os.getcwd(), "classifiers", "round_passed_real_cases")
            if hash_value in self.false_positives:
                self.printer.log('false positive')
                return False
            elif hash_value in self.true_positives:
                self.printer.log('true positive')
                with open("{}/{}.finished".format(direc, hash_value), 'w') as f:
                    f.write('true positive')
                return True
            else:
                image.save("{}/{}.png".format(direc, hash_value))
                finish_line = bool(self.classifier['lap2'].predict(img_array) == [1])

        return finish_line

    def predict_finishline_from_filename(self, filename):
        """
        Predicts if the finish line was passed based on the appearance of the ghost
        :param filename: string with the image filename
        :return: bool, True if finish line was passed, False if not
        """

        image = Image.open(filename).convert('L')
        return self.predict_finishline(image)

    def run_one_round(self, server, method, input_values, multiprocess=None):
        """
        Runs one round, i.e. until the ghost holding the lap sign appears in a screenshot
        :param server: string, either http, socket or mmf
        :param method: string, either
        :param input_values:
        :param multiprocess: boolean, whether the run happens in a multiprocess environment
        :return:
        """
        valid_servers = ('http', 'socket', 'mmf')
        if server.lower() not in valid_servers:
            raise ValueError('Server {} must be in {}'.format(server, valid_servers))

        valid_methods = ('hash', 'nn', 'replay')
        if method.lower() not in valid_methods:
            raise ValueError('Method {} must be in {}'.format(method, valid_methods))

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
                try:
                    img = Image.open(io.BytesIO(buf)).convert('L')
                except OSError as e:
                    self.printer.log(str(e, index))
                    buf += server_receive()
                    img = None
                # check if round passed
                if self.predict_finishline(img):
                    finished = True
                    break
                try:
                    resp = input_values[index]
                except:
                    break
                not_send = 10
                while not_send > 0 and resp is not None:
                    try:
                        server_send(resp)
                        not_send = 0
                    except:
                        self.printer.log('failed')
                        not_send -= 1
                index += 1

        if finished:
            running_time = self.running_time_to_seconds(self.predict_running_time(Image.open(io.BytesIO(buf)).convert('L')))
        else:
            running_time = 0

        if multiprocess is not None:
            multiprocess[self.id] = running_time
        return running_time

    def predict_running_time(self, image, output=False):
        """
        Predicts the running time fron an image, i.e. the time in the right upper corner
        :param image: PIL image
        :param output: boolean, true shows the image in an IPython notebook
        :return: string with the joined digits, not converted to a "real" time
        """
        crop_numbers = (176, 7, 242, 21)
        crop_digits = [0, 8, 24, 32, 48, 56]
        main_img = image.crop(crop_numbers)
        index = 0
        prediction = []
        for i, x in enumerate(crop_digits):
            index += 1
            c = main_img.crop((x, 0, x + 8, 14))
            if output:
                try:
                    IPython.display.display(c)
                except ImportError:
                    warnings.warn('failed to display image, probably Ipython was not imported')

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
        """

        :param image_obj:
        :param run_time:
        :return:
        """

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
                    self.finish_imgs = []
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
        :param image_hash: an image hash
        :return: a byte-string with joypad input
        """
        if self.hashes is None:
            self.calculate_hashes()

        possibilities = self.hashes.get(str(image_hash))
        if not possibilities or len(possibilities) == 0:
            min_hash = min(self.all_hashes, key=lambda x: abs(x - image_hash))

            # slower but deterministic
            if deterministic:
                possibilities = []
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
        :param image:
        :return:
        """
        if self.image_method == 'hash':
            return self.img_hash(image)

        return self.default_joypad()

    def default_joypad(self, player=1):
        """

        :param player:
        :return:
        """
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
        #try:
        image = image.crop(self.crop_box)
        #except:
        #    image = Image.new('RGB', (self.crop_box[2] - self.crop_box[0], self.crop_box[3] - self.crop_box[1]))
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
            filename = os.path.join(os.getcwd(), 'movies_images/new/new_{}.png'.format(self.new_index))
            while os.path.isfile(filename):
                self.new_index += 1
                filename = os.path.join(os.getcwd(), 'movies_images/new/new_{}.png'.format(self.new_index))

            image.save(filename)
            filename = os.path.join(os.getcwd(), 'movies_images/new/new_{}.hash'.format(self.new_index))
            with open(filename, 'wb') as f:
                pickle.dump(image_hash, f, protocol=pickle.HIGHEST_PROTOCOL)
            #filename will be used later
            filename = os.path.join(os.getcwd(), 'movies_images/new/new_{}.txt'.format(self.new_index))

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

    def calculate_hashes(self, overwrite=False,
                         filename_hashes='hashes/hashes.pickle',
                         filename_hash_to_file='hashes/hash_to_file.pickle'):
        """

        :return:
        """
        self.hashes = collections.defaultdict(list)
        self.hash_to_file = collections.defaultdict(list)

        # check if hashes already exist and load them
        if not overwrite and os.path.isfile(filename_hashes):
            start_time = time.time()
            with open(filename_hashes, 'rb') as f:
                self.hashes = pickle.load(f)
            with open(filename_hash_to_file, 'rb') as f:
                self.hash_to_file = pickle.load(f)
            pickled = True

            self.printer.log('done pickling hashes')
            self.printer.log('Time to read pickle: {}'.format(time.time() - start_time))
        else:
            pickled = False

        # get a list of all pressed_keys
        start_time = time.time()
        for folder in os.listdir(os.path.join(self.directory, 'movies_images/')):
            if os.path.isdir(os.path.join(self.directory, 'movies_images/', folder)):
                if not folder.endswith('.bk2'):
                    continue
                filename = os.path.join(self.directory, 'movies_images/', folder, folder)
                if os.path.isfile(filename):
                    m = MovieFile(filename=filename)
                    self.pressed_keys[os.path.basename(filename)] = m.pressed_keys
        self.printer.log('Time for parsing all keys: {}'.format(time.time() - start_time))

        # calculate the hash for all images
        self.all_hashes = []
        start_time = time.time()
        if not pickled and not overwrite:
            for root, dirs, files in os.walk(os.path.join(self.directory, 'movies_images/')):
                for file in files:
                    if file.endswith(('.png', '.bmp', '.jpg')):
                        movie_name = file[0:file.rfind('.bk2') + 4]
                        # skip everything which is not from a movie file
                        if movie_name == 'new' or not root.endswith('.bk2'):
                            continue
                        file_ending = file.split('.')[-1]
                        index = int(file[file.rfind('_') + 1:file.rfind('.{}'.format(file_ending))])
                        image_hash = self.calculate_img_hash(os.path.join(root, file))
                        self.all_hashes.append(image_hash)
                        self.hashes[str(image_hash)].append(self.pressed_keys[movie_name][index])
                        self.hash_to_file[image_hash].append(file)
        self.printer.log('Time for all hashes: {}'.format(time.time() - start_time))

        # if pickled we need to store all image hashes
        if len(self.all_hashes) == 0:
            self.all_hashes = [imagehash.hex_to_hash(i) for i in self.hashes.keys()]

        # pickle the hashes
        if not pickled or overwrite:
            with open(filename_hashes, 'wb') as f:
                pickle.dump(self.hashes, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(filename_hash_to_file, 'wb') as f:
                pickle.dump(self.hash_to_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def filter_keys(pressed_keys, allowed=b'|..|UDLRs.YBXAlr|............|'):
        """
        TO DO: really required?!
        :param pressed_keys:
        :param allowed:
        :return:
        """
        if isinstance(pressed_keys, str) or isinstance(pressed_keys, bytes):
            pressed_keys = [pressed_keys]
            string_input = True
        else:
            string_input = False

        for p, pressed in enumerate(pressed_keys):
            for i, button in enumerate(pressed):
                if button not in ('.', '|') and button != allowed[i:i + 1]:
                    pressed_keys[p] = pressed[0:i] + allowed[i:i + 1] + pressed[i + 1:]
        if string_input:
            return pressed_keys[0]
        else:
            return pressed_keys
