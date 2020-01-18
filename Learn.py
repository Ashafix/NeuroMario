import os
import pathlib
import socket
import hashlib
import pickle
from GameServer import GameServer
from Emuhawk import Emuhawk
from Joypad import get_missing_values
import utils


class Learn:
    def __init__(self, model, weights_index, args, best_score=21.19, predict_finishline=None):
        self.best_score = best_score
        self.model = model
        self.initial_weights = model.get_weights()
        self.weights_index = weights_index
        self.initial_shape = self.initial_weights[weights_index].shape
        self.state = args.state
        if predict_finishline:
            self.predict_finishline = predict_finishline
        else:
            self.predict_finishline = args.predict_finishline
        self.args = args
        if self.state is None:
            self.folder = 'weights'
        else:
            self.folder = 'weights_{}'.format(self.state)
        if not os.path.isdir(os.path.join(os.getcwd(), self.folder)):
            pathlib.Path(os.path.join(os.getcwd(), self.folder)).mkdir(parents=True, exist_ok=True)

    def learn(self, weights, keep_hash=True):
        """

        :param weights:
        :param keep_hash:
        :return:
        """

        _weights = self.model.get_weights()
        if keep_hash:
            adjusted_weights = self.initial_weights[self.weights_index] - weights.reshape(self.initial_shape)
        else:
            adjusted_weights = weights
        _weights[self.weights_index] = adjusted_weights.reshape(self.initial_shape)

        if keep_hash:
            hash_digest = hashlib.sha1(_weights[self.weights_index]).hexdigest()
            filename = '{}/{}.txt'.format(self.folder, hash_digest)
            if os.path.isfile(filename):
                with open(filename, 'r') as f:
                    score = float(f.read().strip())
                    print(filename)
                    print('retrieving hashed score: {}'.format(score))
                    return score

        self.model.set_weights(_weights)
        g = GameServer(socket_autostart=True,
                       socket_ip=socket.gethostbyname(socket.gethostname()),
                       socket_port=9000 + os.getpid() % 1000)
        e = Emuhawk(socket_ip=g.server.ip,
                    socket_port=g.server.port,
                    lua_script=self.args.lua,
                    config_file=self.args.bizhawk_config)
        e.state = self.state
        e.start()
        g.server.create_connection(timeout=60)
        missing = get_missing_values(self.model)
        if self.predict_finishline:
            predictor_finish_line = None
            on_img_error = None
        else:
            predictor_finish_line = lambda _: False
            on_img_error = utils.msg_to_time

        x = g.ai(method='socket',
                 model=self.model,
                 missing=missing,
                 threshold=self.args.play_threshold,
                 bit_array=self.args.bit_array,
                 predictor_finish_line=predictor_finish_line,
                 on_img_error=on_img_error)

        if keep_hash:
            with open('{}/{}.txt'.format(self.folder, hash_digest), 'w') as f:
                f.write(str(x))
            with open('{}/{}.weights-{}'.format(self.folder, hash_digest, self.weights_index), 'wb') as f:
                pickle.dump(_weights[self.weights_index], f)

        print(x)
        return x