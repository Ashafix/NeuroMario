import socket
import time
import datetime
from PIL import Image
import io
import numpy as np
import pickle
from Printer import Printer, PrinterDummy
from GameServer import GameServer


class ServerSocket:
    def __init__(self, ip=None, port=None, printer=None):

        # try to get the IP automatically
        if ip is None:
            self.ip = socket.gethostbyname(socket.gethostname())
        else:
            self.ip = ip

        # let the OS pick the port unless explicitly specified
        if port is None:
            self.port = 0
        else:
            self.port = port
        self.serversocket = None
        self.connection = None
        self.address = None
        if printer is None:
            self.printer = Printer()
        elif not printer or printer == 'dummy':
            self.printer = PrinterDummy()
        else:
            self.printer = printer
        self.img = None
        self.game_server = GameServer(http_autostart=False, socket_autostart=False, verbose=False)

    def __str__(self):
        resp = '{}: ip: {}, port: {}'.format(type(self).__name__, self.ip, self.port)
        return resp

    def __repr__(self):
        return self.__str__()

    def receive(self, image=False, repeats=100, packet_size=16384):

        repeats = repeats
        buf = b''
        while image or repeats > 0:
            try:
                buf += self.connection.recv(packet_size)
            except ConnectionResetError:
                print('Connection was reset during receive')
                return buf
            repeats -= 1
            if len(buf) > 0:
                if not image or buf[-1] == 130:
                    break
        return buf

    def send(self, message, retries=10):
        """
        Sends a message to the client
        :param message: bytes, the message to be send
        :param retries: int, how many times the server tries to send the message
        :return: boolean, True is sent succesful, false if not
        """
        while retries > 0:
            try:
                self.connection.send(message)
                return True
            except:
                retries -= 1

        return False

    def create_connection(self, no_of_connections=10, timeout=10):
        """

        :return:
        """	
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.settimeout(timeout)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.serversocket.bind((self.ip, self.port))
        except OSError:
            self.serversocket.close()
            self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.serversocket.settimeout(timeout)
            self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.serversocket.bind((self.ip, self.port))

        self.serversocket.listen(no_of_connections)
        self.connection, self.address = self.serversocket.accept()

    def listen(self, run_time=600):

        finished = False
        img = b''
        hash_repeat = 0
        new_hash = ''
        last_hash = ''
        start_time = time.time()
        while time.time() - start_time < run_time:
            try:
                buf = self.connection.recv(8192)
            except:
                buf = b''
            if len(buf) > 0:
                if b'PNG' in buf:
                    img = buf
                    finished = False
                else:
                    img += buf
                if buf[-1] == 130:
                    finished = True

            if finished and img != b'':
                finished = False

                self.printer.log('\n###')
                self.printer.log(datetime.datetime.utcnow().strftime('%M:%S.%f')[:-3])

                self.img = img
                try:
                    image_obj = Image.open(io.BytesIO(img))
                    self.calculate_img_hash(image_obj)
                except:
                    image_obj = None
                if image_obj is not None:
                    try:
                        new_hash = str(self.calculate_img_hash(image_obj.crop((0, 25, 256, 224))))
                    except:
                        pass
                    if new_hash == last_hash:
                        hash_repeat += 1
                    else:
                        last_hash = new_hash
                        hash_repeat = 0
                    self.printer.log(datetime.datetime.utcnow().strftime('%M:%S.%f')[:-3])
                    try:
                        resp = self.game_server.img_to_joypad(image_obj)
                    except:
                        resp = bytes(self.game_server.default_joypad(), 'utf-8')
                        self.hashes[new_hash].append(resp)

                    if hash_repeat > 500:
                        resp = b'Restart'

                    not_send = 1000
                    while not_send > 0:
                        try:
                            self.connection.send(resp)
                            not_send -= 1
                        except:
                            self.connection, self.address = self.serversocket.accept()
                    self.connection, self.address = self.serversocket.accept()
                    img = b''

                    if resp == b'Restart':
                        return False
        return True

    def receive_img(self, timeout=10):

        start_time = time.time()
        finished = False
        img = b''
        while (time.time() - start_time) < timeout:
            try:
                buf = self.connection.recv(8192)
            except:
                buf = b''
            if len(buf) > 0:
                if b'PNG' in buf:
                    img = buf
                    finished = False
                else:
                    img += buf
                if buf[-1] == 130:
                    finished = True

            if finished and img:
                self.img = img

                try:
                    # for broken streams which deliver partial PNGs
                    image_obj = Image.open(io.BytesIO(img))
                    img_hash = self.game_server.calculate_img_hash(image_obj)
                    image_gray = image_obj.convert('L')
                except:
                    image_obj = None
                    image_gray = None
                    img_hash = ''

                    self.printer.log('broken png')

                return image_obj, img_hash, image_gray

    def speed_listen(self, verbose=False, run_time=600):

        printer = Printer(verbose=verbose)
        start_time = time.time()
        last_hash = None
        fails = 0
        hash_repeat = 0
        finish_line = False
        cloud = False
        new_hash = ''
        resp = b''

        with open('classifier_lap_rf.txt', 'rb') as f:
            self.classifier['lap'] = pickle.load(f)

        with open('classifier_runingtime.txt', 'rb') as f:
            self.classifier['runningtime'] = pickle.load(f)

        self.connection, self.address = self.serversocket.accept()
        self.connection.send(b'Restart')

        while (time.time() - start_time) < run_time and fails < 3:
            self.connection, self.address = self.serversocket.accept()
            image_obj, img_hash, image_gray = self.receive_img(timeout=10)

            if image_obj and img_hash:
                self.image_obj = image_obj
                try:
                    new_hash = str(self.game_server.calculate_img_hash(image_obj.crop((0, 25, 256, 224))))
                except:
                    pass
                if new_hash == last_hash:
                    hash_repeat += 1
                else:
                    last_hash = new_hash
                    hash_repeat = 0

                if index > 500:
                    if self.classifier['lap'].predict(np.array(image_gray.crop((0, 0, 250, 100))).reshape(1, -1)) == [1]:
                        if not cloud:
                            printer.log('I saw a cloud')
                            r_time = self.predict_running_time(image_gray)
                            self.run_times[-1].append(r_time)
                            printer.log(r_time)
                            hash_repeat = 10 ** 6
                        cloud = True
                    else:
                        printer.log('I stopped seeing clouds')
                        cloud = False
                if str(img_hash) in self.hashes_finished:
                    printer.log('yay, I finished')
                    if not finish_line:
                        self.finish_imgs = list()
                        finish_line = True
                        t = time_from_image()

                if finish_line:
                    if t.image_has_total(np.array(image_obj)):
                        printer.log('total')
                        self.finish_imgs.append(np.array(image_obj))
                        if len(self.finish_imgs) > 100:
                            printer.log(t.predict_time_from_filenames(self.finish_imgs))
                            return True

                if hash_repeat > 500:
                    fails += 1
                    resp = b'Restart'
                else:
                    resp = self.game_server.hash_to_joypad(img_hash, learn=True, allow_random=False)
                not_send = 1000
                while not_send > 0:
                    try:
                        self.connection.send(resp)
                        self.connection.shutdown()
                        self.connection.close()
                        not_send = 0
                    except:
                        # self.connection, self.address = self.serversocket.accept()
                        not_send -= 1
                if not_send > 0:
                    try:
                        self.connection.shutdown()
                        self.connection.close()
                    except:
                        pass

            if hash_repeat == 10 ** 6:
                printer.log('finished by seeing a cloud')
                return True
            elif resp == b'Restart':
                printer.log('failed due to repeats')
                return False
        printer.log('failed due to timeout')
        return False
