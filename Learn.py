import os
import pathlib
import socket
import hashlib
import pickle
from GameServer import GameServer
from Emuhawk import Emuhawk
import utils


class Learn:
    def __init__(self, model, weights_index, args, best_score=21.19):
        self.best_score = best_score
        self.model = model
        self.initial_weights = model.get_weights()
        self.weights_index = weights_index
        self.initial_shape = self.initial_weights[weights_index].shape
        self.state = None
        self.args = args

    def learn(self, weights, keep_hash=True):
        """

        :param weights:
        :param keep_hash:
        :return:
        """
        print(weights)
        if self.state is None:
            folder = 'weights'
        else:
            folder = 'weights_{}'.format(self.state)
        if not os.path.isdir(os.path.join(os.getcwd(), folder)):
            pathlib.Path(os.path.join(os.getcwd(), folder)).mkdir(parents=True, exist_ok=True)

        _weights = self.model.get_weights()
        if keep_hash:
            adjusted_weights = self.initial_weights[self.weights_index] - weights.reshape(self.initial_shape)
        else:
            adjusted_weights = weights
        _weights[self.weights_index] = adjusted_weights.reshape(self.initial_shape)
        self.model.set_weights(_weights)
        if keep_hash:
            hash_digest = hashlib.sha1(_weights[self.weights_index]).hexdigest()
            filename = '{}/{}.txt'.format(folder, hash_digest)
            if os.path.isfile(filename):
                with open(filename, 'r') as f:
                    score = float(f.read().strip())
                    print(filename)
                    print('retrieving hashed score: {}'.format(score))
                    return score
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
        missing = utils.get_missing_values(self.model)

        x = g.ai(method='socket',
                 model=self.model,
                 missing=missing,
                 threshold=self.args.play_threshold,
                 bit_array=self.args.bit_array,
                 predictor_finish_line=lambda x: False,
                 on_img_error=utils.msg_to_time)

        if keep_hash:
            with open('{}/{}.txt'.format(folder, hash_digest), 'w') as f:
                f.write(str(x))
            with open('{}/{}.weights-{}'.format(folder, hash_digest, self.weights_index), 'wb') as f:
                pickle.dump(_weights[self.weights_index], f)

        print(x)
        return x