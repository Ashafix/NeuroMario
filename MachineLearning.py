import os
import numpy as np
from PIL import Image
#import imagehash
import natsort
import keras.models
import keras.optimizers
from keras.layers import Dense, Dropout, Flatten, Conv2D
import pickle
import sklearn.utils
import sklearn.svm
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
from MovieFile import MovieFile
from Joypad import Joypad
try:
    import plotly
except ImportError:
    print("Plotly was not imported, no statistics visualization available")
try:
    import win_unicode_console
    win_unicode_console.enable()
except ImportError:
    print("Keras might throw weird console errors, see https://stackoverflow.com/questions/47356993/oserror-raw-write-returned-invalid-length-when-using-print-in-python")


class MachineLearning:
    """
    Class making all machine learning utility functions for Mario Kart available
    """

    def __init__(self, images=None, filename_joypad=None):
        self.crop_box_time = (145, 0, 242, 24)
        self.crop_box_lap_classifier = (0, 0, 250, 100)
        #self.zero_values = (1046243, # time 00:00:00 taken from Super Mario Kart (USA).bk2_frame_1509.png
        #                    724995)  # time 00:00:00 taken from Super Mario Kart (USA)-Track1.bk2_frame_1206.png
        self.zero_values = (3139390632787292812, -1550348942165723893)
        self.black_bar1 = None  # used for gray screenshots from EmuHawk which don't have a alpha channel
        self.black_bar3 = None  # used for screenshots from EmuHawk which don't have a alpha channel
        self.black_bar4 = None  # user for PNG files which have an alpha channel
        self.create_black_bar()
        self.input = None
        self.output = None
        self.possibilities = set()
        self.model = None

        first_image = None
        if images is None:
            self.image_files = list()
        elif isinstance(images, list):
            self.image_files = images
        elif isinstance(images, str):
            if images.endswith('.png'):
                self.image_files = [images]
            else:
                self.image_files = list()
                first_image = self.add_images_from_dir(images)

        if filename_joypad is None:
            self.pressed_keys = list()
        elif isinstance(filename_joypad, str):
            if first_image is None:
                self.pressed_keys = MovieFile.parse_log_file(filename_joypad)
            else:
                self.pressed_keys = MovieFile.parse_log_file(filename_joypad)[first_image[0]:]
        else:
            self.pressed_keys = filename_joypad

    def add_dir(self, directory):
        """
        Adds images and joypad input from a directory
        :param directory: string, directory which contains the required files
        :return: None
        """

        image_endings = ('.png',)
        files = os.listdir(directory)
        accepted_files = list()
        for f in files:
            if f.endswith(image_endings):
                accepted_files.append(os.path.join(directory, f))

        # make sure they are in the right order since the numbers are not padded
        accepted_files = natsort.natsorted(accepted_files)

        first_img = self.find_first_image(accepted_files)
        last_img = self.find_last_image(accepted_files)
        print('adding directory {} starting from {} to {}'.format(directory, first_img[1], last_img[1]))
        if 'Input Log.txt' in files:
            filename_joypad = 'Input Log.txt'
        else:
            for f in files:
                if f.endswith('.bk2'):
                    filename_joypad = f
                    break
        pressed_keys = MovieFile.parse_log_file(os.path.join(directory, filename_joypad))
        if len(pressed_keys) != len(accepted_files):
            raise ValueError('different number of input and output files')

        self.image_files.extend(accepted_files[first_img[0]:last_img[0] + 1])
        self.pressed_keys.extend(pressed_keys[first_img[0]:last_img[0] + 1])

    def add_images_from_dir(self, directory):
        """

        :param directory:
        :return:
        """

        image_endings = ('.png',)
        files = os.listdir(directory)
        accepted_files = list()
        for f in files:
            if f.endswith(image_endings):
                accepted_files.append(os.path.join(directory, f))
        first_img = self.find_first_image(accepted_files)
        self.image_files.extend(accepted_files[first_img[0]:])
        return first_img

    def find_first_image(self, files, start=1000):
        """
        finds the first image which looks like as if the track has just been loaded
        :param files: list, image files with screenshots
        :param start: int, offset from where the array is searched, should be close to expected first image
        :return: tuple, index and the filename of the first which looks like the start of the race, None if no image was found
        """

        len_files = len(files)
        for i in range(start, len_files + start):
            f = files[i % len_files]
            image = np.array(Image.open(f).crop(self.crop_box_time)).flatten()
            np_hash = hash(tuple(image))
            if np_hash in self.zero_values:
                return i + start, f

    def find_last_image(self, files):
        """
        finds the first image where the ghost holding the finish flag appears,
        then substracts 128 to get the file which represents passing finish line
        :param files: list, image files with screenshots
        :return: tuple, index and the filename, last index and filename if no match was found
        """

        filename = os.path.join(os.getcwd(), 'classifiers/Super Mario Kart (USA)_MarioCircuitI_2.bk2_frame_6371.png')
        ghost = np.array(Image.open(filename).convert('L').crop((81, 12, 121, 48)))
        best = 10**6
        best_file = ""
        for f in files[::-1]:
            a = np.array(Image.open(f).convert('L').crop((81, 12, 121, 48)))
            score = np.sum(np.absolute(ghost - a))
            if score <= 91278:
                if best > score and best != 10**6:
                    break
                best = score
                best_file = f
        if best_file == "":
            return len(files) - 1, files[-1]
        else:
            return files.index(best_file) - 128, best_file

    def create_black_bar(self):
        """
        Defines the dimensions of a black bar used to mask time in images
        :return: None
        """
        if self.black_bar1 is None:
            self.black_bar1 = np.zeros(
                (self.crop_box_time[3] - self.crop_box_time[1], self.crop_box_time[2] - self.crop_box_time[0]))
        if self.black_bar3 is None:
            self.black_bar3 = np.zeros(
                (self.crop_box_time[3] - self.crop_box_time[1], self.crop_box_time[2] - self.crop_box_time[0], 3))
        if self.black_bar4 is None:
            self.black_bar4 = np.zeros(
                (self.crop_box_time[3] - self.crop_box_time[1], self.crop_box_time[2] - self.crop_box_time[0], 4))
            self.black_bar4[..., 3] = 255


    def prepare_image(self, image, black_bar=True, player=1, gray=True, normalize=False, pad_to=None):
        """

        :param image: either a String with the filename or something which can be read by np.array, e.g. a PIL image
        :param black_bar: boolean, whether the time should be blacked out
        :param player: int, 1 if the upper half should be used,
                            2 if the lower half should be used,
                            0 if the full image should be used
        :param gray: boolean, whether to convert to gray scales
        :param normalize: boolean, whether the images are divided by 255 to make them fit in [0, 1]
        :return: numpy array with the adjusted image
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
        img_arr = np.array(img)
        if len(img_arr.shape) < 1:
            raise ValueError('Odd image: {}'.format(img_arr))

        if black_bar:
            if img_arr.shape[-1] == 3:
                img_arr[self.crop_box_time[1]:self.crop_box_time[3],
                self.crop_box_time[0]:self.crop_box_time[2]] = self.black_bar3
            elif img_arr.shape[-1] == 4:
                img_arr[self.crop_box_time[1]:self.crop_box_time[3],
                self.crop_box_time[0]:self.crop_box_time[2]] = self.black_bar4
            elif img_arr.shape[-1] == 256 and not gray:
                img_arr[self.crop_box_time[1]:self.crop_box_time[3],
                self.crop_box_time[0]:self.crop_box_time[2]] = self.black_bar1
            else:
                raise ValueError('Color channel must be either 3 or 4, but found {}'.format(img_arr.shape[-1]))

        if player > 0:
            y_start = (player - 1) * int(img_arr.shape[0] / 2)
            y_end = player * int(img_arr.shape[0] / 2)
            img_arr = img_arr[y_start:y_end, 0:img_arr.shape[1]]

        if gray:
            img_arr = np.array(Image.fromarray(img_arr).convert('L'))

        elif img_arr.shape[-1] == 4:
            img_arr = img_arr[:,:,0:3]

        if normalize:
            img_arr = img_arr / 255
        if pad_to is not None:
            if len(pad_to) == 2 and not gray:
                pad_to += (img_arr.shape[-1], )
            padded = np.zeros(pad_to)
            max_x = min(img_arr.shape[0], padded.shape[0])
            max_y = min(img_arr.shape[1], padded.shape[1])
            padded[:max_x, :max_y] = img_arr[:max_x, :max_y]
            return padded
        
        return img_arr

    def input_output(self, black_bar=True, player=1, gray=True, normalize=False, mirror=False, bit_array=False,
                     pickle_files=None, shuffle=True, drop_unused_columns=True, random_state=42, pad_to=None):
        """
        Converts images and joypad input to numpy arrays
        :param black_bar: whether a black bar should put over the time indicator
        :param player: int, whether the upper (1) or lower half (2) of the image should be used
        :param gray: boolean, whether images should be converted to gray scale
        :param normalize: boolean, whether the input array should be normalized to be between 0 and 1
        :param mirror: boolean, whether images where only the left or right button is pressed should be mirrored
        :param bit_array: boolean, whether the output should be translated to a bit array with max one true value,
                          i.e. squash inputs, passed to create_output_array
        :param pickle_files: tuple/list, where to store the pickled input and output files
        :param shuffle: boolean, whether the input is shuffled or not
        :param random_state: int, random_state for sklearn.shuffle
        :return: tuple, input and output array
        """

        if pickle_files is not None:
            if len(pickle_files) != 2:
                raise ValueError('pickle_files needs to be a tuple/list with len == 2, got: '.format(pickle_files))
            if os.path.isfile(pickle_files[0]) and os.path.isfile(pickle_files[1]):
                print('Using pickled files')
                with open(pickle_files[0], 'rb') as f:
                    self.input = np.load(f)
                with open(pickle_files[1], 'rb') as f:
                    self.output = np.load(f)
                return self.input, self.output

        if self.image_files is None or self.pressed_keys is None or len(self.image_files) == 0:
            raise ValueError('At least one image and joypad input is required')

        if len(self.image_files) != len(self.pressed_keys):
            raise ValueError('Number of images does not match length of joypad input')

        if gray:
            input_shape = np.array(Image.open(self.image_files[0]).convert('L')).shape[0:2]
        else:
            input_shape = np.array(Image.open(self.image_files[0])).shape
            # remove alpha channel
            if input_shape[-1] == 4:
                input_shape = input_shape[0:-1] + (3, )

        # take only upper or lower half of the image
        input_shape = (input_shape[0] // 2, ) + input_shape[1:]
        
        if normalize:
            input_dtype = np.float32
        else:
            input_dtype = np.uint8
        
        if pad_to is None:
            input_array = np.zeros((len(self.image_files), ) + input_shape,
                                   dtype=input_dtype)
        else:
            input_array = np.zeros((len(self.image_files), ) + pad_to,
                                   dtype=input_dtype)

        for i, image in enumerate(self.image_files):
            input_array[i] = self.prepare_image(image, black_bar=black_bar, gray=gray, pad_to=pad_to)

        output_array = self.create_output_array(player)
        if mirror:
            self.input, self.output = self.mirror_images(input_array, output_array[0], player=player)
        else:
            self.input = input_array
            self.output = output_array[0]

        if drop_unused_columns:
            idx = np.argwhere(np.all(self.output[..., :] == 0, axis=0))
            self.output = np.delete(self.output, idx, axis=1)

        if bit_array:
            # works only in numpy >v.1.12
            #possible_bits = np.unique(self.output, axis=0)
            hashable = map(tuple, self.output)
            possible_bits = list(set(hashable))
            possible_bits.sort()
            possible_bits = np.array(possible_bits)
            print(possible_bits)

            output_bits = np.zeros((self.output.shape[0], possible_bits.shape[0]), dtype=np.uint8)

            # TODO there must a more efficient way than two loops
            for i in range(output_bits.shape[0]):
                for j in range(possible_bits.shape[0]):
                    if np.all(possible_bits[j] == self.output[i]):
                        output_bits[i][j] = 1
                        break
            self.output = output_bits

        if shuffle:
            self.input, self.output = sklearn.utils.shuffle(self.input, self.output, random_state=random_state)
        if pickle_files is not None:
            np.save(pickle_files[0], self.input)
            np.save(pickle_files[1], self.output)
        return self.input, self.output

    def mirror_images(self, input_array, output_array, player=1):
        """
        Mirrors images if either up or down, or left or right button is pressed
        :return: None
        """

        if output_array.shape[0] != len(self.pressed_keys) or input_array.shape[0] != len(self.image_files):
            raise ValueError('Shape mismatch, mirror_images() can be only called once')

        start = 4
        up = Joypad.up[start:start + 12].replace(b'.', b'')
        down = Joypad.down[start:start + 12].replace(b'.', b'')
        right = Joypad.right[start:start + 12].replace(b'.', b'')
        left = Joypad.left[start:start + 12].replace(b'.', b'')
        start = (player - 1) * 13 + 4

        to_be_mirrored = list()
        for i, pressed in enumerate(self.pressed_keys):
            p = pressed[start:start + 12]
            if up in p and down not in p:
                to_be_mirrored.append([i, 'up'])
            elif down in p and up not in p:
                to_be_mirrored.append([i, 'down'])
            if left in p and right not in p:
                to_be_mirrored.append([i, 'left'])
            elif right in p and left not in p:
                to_be_mirrored.append([i, 'right'])

        new_input = np.zeros(((len(self.image_files) + len(to_be_mirrored),) + input_array.shape[1:]),
                             dtype=np.uint8)
        new_output = np.zeros((len(self.pressed_keys) + len(to_be_mirrored), 12),
                              dtype=np.uint8)
        new_input[0:len(self.image_files)] = input_array
        new_output[0:len(self.pressed_keys)] = output_array

        offset = len(self.image_files)
        translation = {'up': 'down',
                       'down': 'up',
                       'left': 'right',
                       'right': 'left'}
        for i, mirror in enumerate(to_be_mirrored):
            new_input[offset + i] = self.flip_image(input_array[i], translation[mirror[1]])
            new_output[offset + i] = Joypad.flip_input(output_array[mirror[0]])

        return new_input, new_output

    @staticmethod
    def flip_image(image_array, rotation):
        """
        Rotates an image array along the x or y-axis
        :param image_array: Numpy array with the image
        :param rotation: either string (up, down, right, left) or int (0, 1, 2, 3)
        :return:
        """

        if isinstance(rotation, str):
            rotation_axis = ('up', 'down', 'left', 'right').index(rotation)
        else:
            rotation_axis = rotation
        if rotation_axis not in (0, 1, 2, 3):
            raise ValueError('unknown rotation: {}'.format(rotation))

        axis = rotation_axis // 2
        return np.flip(image_array, axis)

    def create_output_array(self, player=1, bit_array=False):
        """

        :param player:
        :param bit_array:
        :return:
        """

        empty_input = Joypad.empty

        if player not in (1, 2):
            raise ValueError('Player must be either 1 or 2')
        self.possibilities = set()

        pipes = [empty_input.find(b'|')]
        while empty_input.find(b'|', pipes[-1] + 1):
            pipes.append(empty_input.find(b'|', pipes[-1] + 1))
        pipes.pop(-1)
        for inp in self.pressed_keys:
            if isinstance(inp, bytes):
                inp = inp.decode()
            self.possibilities.add(inp[pipes[player] + 1:pipes[player + 1]])
        output_array = np.zeros((len(self.pressed_keys), pipes[2] - pipes[1] - 1),
                                    dtype=np.uint8)

        for i, inp in enumerate(self.pressed_keys):
            if isinstance(inp, bytes):
                inp = inp.decode()
            self.possibilities.add(inp[pipes[player] + 1:pipes[player + 1]])
            output_array[i] = Joypad.joypad_to_array(inp[pipes[player] + 1:pipes[player + 1]])
        return output_array, self.possibilities

    def neural_net(self, batch_size=50, epochs=100, keep_prob=0.8,
                   modelname='model', loss='mean_squared_error', metrics=None,
                   optimizer=keras.optimizers.adam(), activation='softsign'):
        """
        Builds and trains a neural net
        :param batch_size:
        :param epochs: int, number of epochs
        :param keep_prob: float, used for dropout, probably to keep the neuron
        :param modelname: string, filename prefix, json/weights/history will be appended
        :param loss: string, Keras loss function
        :param metrics:
        :param optimizer: keras.optimizer
        :param activation: string, keras.activation for final layer
        :return: keras.model
        """

        if metrics is None:
            metrics = ['accuracy']

        if modelname is None:
            modelname = 'model_{}_{}_{}'.format(loss,
                                                str(optimizer).split(' ')[0][1:],
                                                activation)

        model = keras.models.Sequential()
        model.add(
            Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=self.input.shape[1:] + (1,)))
        model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(1164, activation='relu'))
        drop_out = 1 - keep_prob
        model.add(Dropout(drop_out))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(drop_out))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(drop_out))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(drop_out))
        model.add(Dense(self.output.shape[1], activation=activation))

        model.compile(loss=loss, optimizer=optimizer,
                      metrics=metrics)
        history = model.fit(self.input.reshape(self.input.shape + (1,)), self.output,
                            batch_size=batch_size, epochs=epochs,
                            shuffle=True, validation_split=0.1)

        model.save_weights('{}_weights.h5'.format(modelname))
        model_json = model.to_json()
        with open('{}.json'.format(modelname), 'w') as json_file:
            json_file.write(model_json)
        with open('{}.history'.format(modelname), 'wb') as history_file:
            pickle.dump(history.history, history_file)
        self.visualize_statistics(history.history, modelname)
        self.model = model
        return model

    @staticmethod
    def remove_empty_columns(array, threshold=0.01):
        """
        Removes empty columns from the output to ease learning
        Assumes that all values are either 0 or 1
        Will give weird results if other values are used
        :param array: Numpy array
        :param threshold: float, if the sum of a column divided the number of columsn is below this threshold,
                          it wioll be dropped
        :return: trimmed Numpy array, index of columns which removed
        """

        if len(array.shape) != 2:
            raise IndexError('Array shape needs to be (samples, output)')

        to_be_dropped = list()
        for col in range(array.shape[1]):
            if array[:, col].sum() / array.shape[0] < threshold:
                to_be_dropped.append(col)

        return np.delete(array, to_be_dropped, 1), to_be_dropped

    def test_neural_net(self):
        """
        Tests whether a neural net can correctly classify three example images
        Prints the statistics for the output buttons
        :return: None
        """

        if not isinstance(self.model, keras.models.Sequential):
            modelname = self.model
            with open('{}.json'.format(modelname), 'r') as f:
                model = keras.models.model_from_json(f.read())
            model.load_weights('{}_weights.h5'.format(modelname))
        else:
            model = self.model
        image = 'movies_images\Super Mario Kart (USA).bk2\Super Mario Kart (USA).bk2_frame_1940.png'
        ml = MachineLearning()
        img = ml.prepare_image(image, normalize=True)
        print('Start: \n', model.predict(img.reshape(1, 112, 256, 1)))
        image = 'movies_images\Super Mario Kart (USA).bk2\Super Mario Kart (USA).bk2_frame_2200.png'
        img = ml.prepare_image(image, normalize=True)
        print('Left: \n', model.predict(img.reshape(1, 112, 256, 1)))
        image = 'movies_images\Super Mario Kart (USA).bk2\Super Mario Kart (USA).bk2_frame_6890.png'
        img = ml.prepare_image(image, normalize=True)
        print('Right: \n', model.predict(img.reshape(1, 112, 256, 1)))

        p = model.predict(self.input.reshape(self.input.shape + (1,)))
        for i in self.output.shape[1]:
            print(np.average(p[:, i]))

    def visualize_statistics(self, history, modelname):
        """
        Visualizes loss and accuracy of Keras model history using Plotly
        :param history: either a string with the filename of the pickled history dict, or the history dict itself
        :return:
        """
        if isinstance(history, str):
            with open('{}.history'.format(history), 'rb') as f:
                history = pickle.load(f)

        graphs_acc = list()
        graphs_loss = list()
        for k in history.keys():
            if 'loss' in k:
                graphs_loss.append(plotly.graph_objs.Scatter(y=history[k], name=k))
            else:
                graphs_acc.append(plotly.graph_objs.Scatter(y=history[k], name=k))

        fig = plotly.tools.make_subplots(rows=1, cols=2)
        for g in graphs_loss:
            fig.append_trace(g, row=1, col=1)
        for g in graphs_acc:
            fig.append_trace(g, row=1, col=2)
        return plotly.offline.plot(fig, filename="{}.html".format(modelname))

    def create_lap_classifier(self, random_state=42, filename=None, test_size=0.2, max_depth=2):
        """
        Creates a RandomForest classifier for recognizing when the player passed the finish line
        :param random_state: int, passed to RandomForestClassifier
        :param filename: str, if not None, filename will be used for pickling
        :return: sklearn.ensemble.RandomForestClassifier
        """

        base_dir = 'number_training/lap/'
        dir_pos = [base_dir + 'positive']
        dir_pos.append(os.path.join(os.getcwd(), 'classifiers/round_passed_real_cases/true_positives'))
        dir_neg = [base_dir + 'negative']
        dir_neg.append('classifiers/fallingDown_Reverse/positive')
        dir_neg.append('classifiers/round_passed_real_cases')

        filenames_pos = [os.path.join(os.getcwd(), dir_pos[0], f) for f in os.listdir(dir_pos[0]) if
                         f.endswith((".png", ".jpg"))]
        for direc in dir_pos[1:]:
            for f in os.listdir(os.path.join(os.getcwd(), direc)):
                if f.endswith((".png", ".jpg")):
                    filenames_pos.append(os.path.join(os.getcwd(), direc, f))

        filenames_neg = [os.path.join(os.getcwd(), dir_neg[0], f) for f in os.listdir(dir_neg[0]) if
                         f.endswith((".png", ".jpg"))]
        for direc in dir_neg[1:]:
            for f in os.listdir(os.path.join(os.getcwd(), direc)):
                if f.endswith((".png", ".jpg")):
                    filenames_neg.append(os.path.join(os.getcwd(), direc, f))

        data = np.zeros((len(filenames_neg) + len(filenames_pos), 100, 250))

        print('start reading images')
        for i, f in enumerate(filenames_neg):
            data[i] = np.array(Image.open(f).crop(self.crop_box_lap_classifier).convert('L'))

        for i, f in enumerate(filenames_pos):
            data[i + len(filenames_neg)] = np.array(Image.open(f).crop(self.crop_box_lap_classifier).convert('L'))

        target = [0 for _ in filenames_neg]
        target.extend([1 for _ in filenames_pos])

        data = data.reshape((len(data), -1))
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)

        print("Fitting the classifier to the training set")
        random_forest_clf = sklearn.ensemble.RandomForestClassifier(max_depth=max_depth,
                                                                    random_state=random_state,
                                                                    verbose=True)
        random_forest_clf.fit(X_train, y_train)
        print(random_forest_clf.score(X_test, y_test))

        if filename is not None and filename.strip() != "":
            with open(filename, 'wb') as f:
                pickle.dump(random_forest_clf, f)
        return random_forest_clf
