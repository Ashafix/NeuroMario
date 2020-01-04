import os
import sys
import argparse
import scipy.optimize as optimize
import pickle
import socket
import keras
import hashlib
import pathlib
from Emuhawk import Emuhawk
from MachineLearning import MachineLearning
from GameServer import GameServer
import utils


class Learn:
    def __init__(self, model, weights_index):
        self.best_score = 21.19
        self.model = model
        self.initial_weights = model.get_weights()
        self.weights_index = weights_index
        self.initial_shape = self.initial_weights[weights_index].shape
        self.state = None

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
            hash_digest = hashlib.sha1(_weights[-2]).hexdigest()
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
                    lua_script=os.path.join(os.getcwd(), 'lua/listen_socket_screenshot.lua'))
        if self.state is None:
            e.state = os.path.join(os.getcwd(), 'states/GhostValley.State')
        else:
            e.state = self.state
        e.start()
        g.server.create_connection(timeout=60)
        missing = get_missing_values(self.model)
        x = g.ai(method='socket',
                 model=self.model,
                 missing=missing,
                 threshold=args.play_threshold,
                 bit_array=True)

        if keep_hash:
            with open('{}/{}.txt'.format(folder, hash_digest), 'w') as f:
                f.write(str(x))
            with open('{}/{}.weights-2'.format(folder, hash_digest), 'wb') as f:
                pickle.dump(_weights[-2], f)
        
        if x != self.best_score:
            xx = g.ai(method='socket',
                      model=self.model,
                      missing=missing,
                      threshold=args.play_threshold,
                      bit_array=True)
            if x != xx:
                print('something went wrong, {} vs {}'.format(x, xx))
            if xx != self.best_score and xx != 100:
                print('new best score: {}'.format(xx))
                self.best_score = xx
                return xx
        print(x)
        return x


def workflow_learn(args, index=-2, seed=42, modelname='model_binary_crossentropy_keras.optimizers.Adam_sigmoid'):
    """

    :param args: argparse object
    :param index: which layer of the network is optimized
    :param seed: int, passed as seed to differential_evolution
    :param modelname: string, the model to load, .json/.weights.h5 is appended automatically
    :return:
    """
    with open('{}.json'.format(modelname), 'r') as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights('{}_weights.h5'.format(modelname))
    learn = Learn(model, index)
    learn.state = args.state
    bounds = list()
    weights = model.get_weights()[index].flatten()
    for i in range(weights.size):
        _min = -abs(weights[i])
        _max = abs(weights[i])
        if _max < 0.1:
            _max = 0.1
            _min = -0.1
        bounds.append((_min, _max))
    optimize.differential_evolution(learn.learn, bounds, seed=seed)


def get_missing_values(model):
    """
    Adds missing joypad values to output
    assumes that if the model produces an output array of size 4 that B, left, right, L are valid options
    assumes that if the model produces an output array of size 3 that B, left, right are valid options
    :param model:
    :return:
    """
    if model.get_weights()[-1].size == 4:
        missing = [0, 1, 4, 5, 6, 8, 9, 11]
    elif model.get_weights()[-1].size == 3:
        missing = [0, 1, 4, 5, 6, 8, 9, 10, 11]
    else:
        raise ValueError("Could not guess which input values are missing based on the output")
    return missing


def find_best_score():
    """

    :return:
    """
    best_score = None
    best_weight = None
    full_dir = os.path.join(os.getcwd(), 'weights')
    all_filenames = os.listdir(full_dir)

    for filename in all_filenames:
        if filename.endswith('.txt'):
            with open(os.path.join(full_dir, filename), 'r') as f:
                new_score = f.read()
                new_score = float(new_score)
            if new_score < best_score:
                new_filename = filename.replace('.txt', '.weights')
                for weights_file in all_filenames:
                    if weights_file.startswith(new_filename):
                        best_score = new_score
                        best_weight = os.path.join(os.getcwd(), 'weights', weights_file)
                        break
    print(best_score, best_weight)
    if best_score is None or best_weight is None:
        raise ValueError('failed to find best score or weight, check if the folder "weights" is not empty')
    return best_score, best_weight


def replay(args, best=True, modelname='model_binary_crossentropy_keras.optimizers.Adam_sigmoid'):
    """

    :param args: argparse object
    :param best: boolean, True, the best model will be played, False, the first model will be played
    :param modelname: string, the name of the model, .json/_weights.h5 will be automatically attached
    :return: None
    """
    
    with open('{}.json'.format(modelname), 'r') as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights('{}_weights.h5'.format(modelname))
    
    if best:
        best_score, best_weight = find_best_score()
        index = int(best_weight.split('weights')[-1])
        weights = model.get_weights()
        with open(best_weight, 'rb') as f:
            weights[index] = pickle.load(f).reshape(weights[index].shape)
        model.set_weights(weights)
    
    g = GameServer(socket_autostart=True,
                   socket_ip=socket.gethostbyname(socket.gethostname()),
                   socket_port=9000 + os.getpid() % 1000)
    e = Emuhawk(socket_ip=g.server.ip,
                socket_port=g.server.port,
                lua_script=os.path.join(os.getcwd(), 'lua/listen_socket_screenshot.lua'))
    e.state = os.path.join(os.getcwd(), args.state)
    e.start()
    g.server.create_connection(timeout=30)
    missing = get_missing_values(model)
    score = g.ai(method='socket',
                 model=model,
                 missing=missing,
                 threshold=args.play_threshold)
    if best:
        assert(score == best_score)


def replay_best(args):
    """
    Replays the best learned model
    :return: None
    """
    replay(args, best=True)


def replay_initial(args):
    """
    Replays the initial neural net work
    :return: None
    """
    replay(args, best=False)


def build_model(args):
    """
    Builds a neural network based on the input files present in movies_images
    :param args:
    :return:
    """
    ml = MachineLearning()
    print('reading input files')
    if args.pickle_files is None or not os.path.isfile(args.pickle_files[0]) or not os.path.isfile(args.pickle_files[1]):
        for direc in os.listdir(os.path.join(os.getcwd(), 'movies_images')):
            dir_name = os.path.join(os.getcwd(), 'movies_images', direc)
            if direc.endswith('bk2') and os.path.isdir(dir_name):
                ml.add_dir(dir_name)
        print(ml.image_files[0])
        print(ml.image_files[-1])

    print('building input and output')
    ml.input_output(normalize=args.normalize, mirror=args.mirror, bit_array=args.bit_array, pickle_files=args.pickle_files)
    print('dropping unused output columns')
    new_output, dropped = ml.remove_empty_columns(ml.output, args.drop_threshold)
    print(dropped)
    ml.output = new_output
    print('training model')
    if args.model is not None:
        modelname = args.model
    else:
        modelname = 'default'
    ml.neural_net(modelname=modelname, activation=args.activation, epochs=args.epochs, loss=args.loss)
    return ml


def play(state, modelname):
    """
    Plays a specific state with a model
    :param state: string, the name of the state which is loaded in Emuhawk
    :param modelname: string, the name of the model is loaded (without .json or .weights)
    :return: None
    """
    with open('{}.json'.format(modelname), 'r') as f:
        model = keras.models.model_from_json(f.read())

    model.load_weights('{}_weights.h5'.format(modelname))
    g = GameServer(socket_autostart=True,
                   socket_ip=socket.gethostbyname(socket.gethostname()),
                   socket_port=9000 + os.getpid() % 1000)
    print('initialized gameserver')
    e = Emuhawk(socket_ip=g.server.ip,
                socket_port=g.server.port,
                lua_script=os.path.join(os.getcwd(), 'lua/listen_socket_screenshot.lua'))
    print('initialized emuhawk')
    if not os.path.isfile(state):
        state = os.path.join(os.getcwd(), state)
    e.state = state
    e.start()
    print('started emuhawk')
    g.server.create_connection(timeout=60)
    missing = get_missing_values(model)
    print('created connection')
    x = g.ai(method='socket',
             model=model,
             missing=missing,
             threshold=args.play_threshold,
             bit_array=True)


def test_model(args):
    """

    :param args: parsed arguments from argparse
    :return: None
    """
    modelname = args.model
    with open('{}.json'.format(modelname), 'r') as f:
        model = keras.models.model_from_json(f.read())

    model.load_weights('{}_weights.h5'.format(modelname))
    ml = MachineLearning()
    ml.model = model
    ml.test_neural_net()


def parse_args(sys_args):
    """
    Parse command line arguments and returns parsed arguments
    :param sys_args: list, arguments coming from sys.argv
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=('learn', 'replay_best', 'replay_initial', 'find_best_score', 'build_model', 'play', 'test', 'clean', 'create_state'))
    parser.add_argument("--model")
    parser.add_argument("--state", default='states/GhostValley_2.State', type=str)
    parser.add_argument("--activation", default="sigmoid")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--loss", default="categorical_crossentropy")
    parser.add_argument("--drop_threshold", default=0.01, type=float)
    parser.add_argument("--play_threshold", default=0.5, type=float)
    parser.add_argument("--normalize", default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--mirror", default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--bit_array", default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--directory", default="weights_states\MarioCircuitI.State")
    parser.add_argument("--suffix", default=".weights-2")
    parser.add_argument("--gametype", default="1P", choices=['1P', '2P'])
    parser.add_argument("--racetype", default="Time Trial")
    parser.add_argument("--player", default="Mario")
    parser.add_argument("--cc_class", default=None, choices=[None, 50, 100, 150])
    parser.add_argument("--cup", default="MUSHROOM CUP")
    parser.add_argument("--track", default="GHOST VALLEY 1")
    parser.add_argument("--filename")
    parser.add_argument("--pickle_files", default=None, nargs=2)
    args = parser.parse_args(sys_args)
    return args


def clean_up(args):
    """

    :param args: argparser object
    :return: None
    """

    print('cleaning')
    utils.delete_all_good_runs(direc=args.directory, suffix=args.suffix)
    utils.write_false_positives()


def create_state(args):
    """
    Calls Emuhawk.create_create_defined_state to create a defined input state
    :param args:
    :return: None
    """

    """
    def create_defined_state(self, gametype='1P', racetype='Time Trial', player='Mario', cc_class=None,
                             cup='MUSHROOM CUP', track='GHOST VALLEY 1',
                             filename_movie=None, auto_start=True):
    """

    g = GameServer(socket_autostart=True,
                   socket_ip=socket.gethostbyname(socket.gethostname()),
                   socket_port=9000 + os.getpid() % 1000)
    e = Emuhawk(socket_ip=g.server.ip,
                socket_port=g.server.port)
    e.create_defined_state(gametype=args.gametype,
                           racetype=args.racetype,
                           player=args.player,
                           cc_class=args.cc_class,
                           cup=args.cup,
                           track=args.track,
                           filename_movie=args.filename)


def main(args):
    """
    Main function used to encapsulated different workflow
    :param args: list, coming from sys.argv
    :return: None
    """

    if args.command == 'learn':
        if args.model is not None:
            workflow_learn(args, modelname=args.model)
        else:
            workflow_learn(args)
    elif args.command == 'replay_best':
        replay_best(args)
    elif args.command == 'replay_initial':
        replay_initial(args)
    elif args.command == 'find_best_score':
        find_best_score()
    elif args.command == 'build_model':
        return build_model(args)
    elif args.command == 'play':
        play(args.state, args.model)
    elif args.command == 'test':
        test_model(args)
    elif args.command == 'clean':
        clean_up(args)
    elif args.command == 'create_state':
        create_state(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args is None:
        sys.exit(1)
    main(args)
