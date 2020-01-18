import os
import sys
import argparse
import scipy.optimize as optimize
import pickle
import socket
from Emuhawk import Emuhawk
from MachineLearning import MachineLearning
from GameServer import GameServer
from Learn import Learn
from Joypad import get_missing_values
import utils


def workflow_learn(args, index=-2, seed=42, modelname='model_binary_crossentropy_keras.optimizers.Adam_sigmoid'):
    """

    :param args: argparse object, overwerites modelname
    :param index: which layer of the network is optimized
    :param seed: int, passed as seed to differential_evolution
    :param modelname: string, the model to load, .json/.weights.h5 is appended automatically
    :return:
    """

    if args.model is not None:
        modelname = args.model
    model = utils.load_model(modelname)
    learn = Learn(model, index, args)
    bounds = []
    weights = model.get_weights()[index].flatten()
    for i in range(weights.size):
        _max = max(abs(weights[i]), 0.1)
        _min = min(-_max, -0.1)

        bounds.append((_min, _max))
    if args.workers == -1:
        args.workers = os.cpu_count()
    optimize.differential_evolution(learn.learn, bounds, seed=seed, workers=args.workers)


def find_best_score(_):
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

    model = utils.load_model(modelname)

    if best:
        best_score, best_weight = find_best_score('')
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
        assert score == best_score, 'Best score {} did not match achieved score'.format(best_score, score)


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
    ml.input_output(normalize=args.normalize, mirror=args.mirror,
                    bit_array=args.bit_array, pickle_files=args.pickle_files)
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
    print('created model: {}'.format(modelname))
    return ml


def play(args=None, state=None, modelname=None):
    """
    Plays a specific state with a model
    :param args: parsed arguments from argparse, will overwrite state and modelname
    :param state: string, the name of the state which is loaded in Emuhawk
    :param modelname: string, the name of the model is loaded (without .json or .weights)
    :return: None
    """

    if args is not None:
        state = args.state
        modelname = args.model
    if state is None or modelname is None:
        raise ValueError('state and modelname (filename of existing model) is required')

    model = utils.load_model(modelname)

    g = GameServer(socket_autostart=True,
                   socket_ip=socket.gethostbyname(socket.gethostname()),
                   socket_port=9000 + os.getpid() % 1000)
    print('initialized gameserver')
    if args.lua in ('', None):
        lua_script = 'lua/listen_socket_screenshot.lua'
    else:
        lua_script = args.lua
    e = Emuhawk(socket_ip=g.server.ip,
                socket_port=g.server.port,
                lua_script=os.path.join(os.getcwd(), lua_script))
    print('initialized emuhawk')
    if not os.path.isfile(state):
        state = os.path.join(os.getcwd(), state)
    e.state = state
    e.start()
    print('started emuhawk')
    g.server.create_connection(timeout=60)
    missing = get_missing_values(model)
    print('created connection')

    return g.ai(method='socket',
                model=model,
                missing=missing,
                threshold=args.play_threshold,
                bit_array=args.bit_array,
                on_img_error=utils.msg_to_time)


def test_model(args):
    """

    :param args: parsed arguments from argparse
    :return: None
    """
    modelname = args.model
    model = utils.load_model(modelname)
    ml = MachineLearning()
    ml.model = model
    ml.test_neural_net()


def replay_with_weights(args):

    model = utils.load_model(args.model)

    with open(args.weights, 'rb') as f:
        weights = pickle.load(f)
    _weights = model.get_weights()

    _weights[-2] = weights
    model.set_weights(_weights)

    g = GameServer(socket_autostart=True,
                   socket_ip=socket.gethostbyname(socket.gethostname()),
                   socket_port=9000 + os.getpid() % 1000)
    print('initialized gameserver')

    e = Emuhawk(socket_ip=g.server.ip,
                socket_port=g.server.port,
                lua_script=os.path.join(os.getcwd(), args.lua))
    print('initialized emuhawk')
    state = os.path.join(os.getcwd(), args.state)
    e.state = state
    e.start()
    print('started emuhawk')
    g.server.create_connection(timeout=60)
    missing = get_missing_values(model)
    print('created connection')

    return g.ai(method='socket',
                model=model,
                missing=missing,
                threshold=args.play_threshold,
                bit_array=args.bit_array,
                on_img_error=utils.msg_to_time)



def parse_args(sys_args):
    """
    Parse command line arguments and returns parsed arguments
    :param sys_args: list, arguments coming from sys.argv
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=('learn', 'replay_best', 'replay_initial', 'find_best_score', 'build_model',
                                            'play', 'test', 'clean', 'create_state', 'replay_with_weights'))
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--state", default='states/GhostValley_2.State', type=str)
    parser.add_argument("--activation", default='sigmoid', type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--loss", default='categorical_crossentropy', type=str)
    parser.add_argument("--drop_threshold", default=0.01, type=float)
    parser.add_argument("--play_threshold", default=0.5, type=float)
    parser.add_argument("--normalize", default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--mirror", default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--bit_array", default=True, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument("--directory", default='weights_states/MarioCircuitI.State', type=str)
    parser.add_argument("--suffix", default='.weights-2', type=str)
    parser.add_argument("--gametype", default='1P', choices=['1P', '2P'])
    parser.add_argument("--racetype", default='Time Trial', type=str)
    parser.add_argument("--player", default='Mario', type=str)
    parser.add_argument("--cc_class", default=None, choices=[None, 50, 100, 150])
    parser.add_argument("--cup", default='MUSHROOM CUP', type=str)
    parser.add_argument("--track", default='GHOST VALLEY 1', type=str)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--pickle_files", default=None, nargs=2)
    parser.add_argument("--workers", default=1, type=str)
    parser.add_argument("--bizhawk_config", default='', type=str)
    parser.add_argument('--lua', default=None, type=str)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--predict_finishline', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
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

    command_to_function = {
        'learn': workflow_learn,
        'replay_best': replay_best,
        'replay_initial': replay_initial,
        'find_best_score': find_best_score,
        'replay_with_weights': replay_with_weights,
        'build_model': build_model,
        'play': play,
        'create_state': create_state
    }

    function = command_to_function[args.command]
    return function(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args is None:
        sys.exit(1)
    print(main(args))
