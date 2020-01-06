import os
from PIL import Image
import os
import shutil
import imagehash
from PIL import Image
import numpy as np
import yaml
import time
import keras.models


def get_false_positives():

    direc = os.path.join(os.getcwd(), 'classifiers', 'round_passed_real_cases')
    files = os.listdir(direc)
    hashes = list()
    for file in files:
        if file.endswith('.png') and not os.path.isfile(os.path.join(os.path.split(file)[0], "true_positives", os.path.split(file)[1])):
            img = Image.open(os.path.join(direc, file))
            hashes.append(str(imagehash.phash(img.crop((0, 0, 250, 100)), hash_size=5)))
        elif file.endswith('.png') and os.path.isfile(os.path.join(os.path.split(file)[0], "true_positives", os.path.split(file)[1])):
            os.remove(file)
    hashes = list(set(hashes))
    print(', '.join(hashes))
    with open('hashes.txt', 'w') as f:
        f.write(', '.join(hashes))

    return hashes


def get_true_positives():

    direc = os.path.join(os.getcwd(), 'classifiers', 'round_passed_real_cases', 'true_positives')
    files = os.listdir(direc)
    hashes = list()
    for file in files:
        if file.endswith('.png'):
            img = Image.open(os.path.join(direc, file))
            hashes.append(str(imagehash.phash(img.crop((0, 0, 250, 100)), hash_size=5)))
    hashes = list(set(hashes))
    print(', '.join(hashes))
    return hashes


def write_false_positives():

    hashes = get_false_positives()
    with open('NeuroMarioConfig.yaml', 'r') as f:
        conf = yaml.load(f)
    conf['finish_line_false_positives'] = hashes
    hashes = get_true_positives()
    conf['finish_line_true_positives'] = hashes
    with open('NeuroMarioConfig.yaml', 'w') as f:
        yaml.dump(conf, f)
    for true_pos in conf['finish_line_true_positives']:
        if true_pos in conf['finish_line_false_positives']:
            raise ValueError('hash collision: {}'.format(true_pos))


def delete_all_good_runs(direc="", suffix=".weights-2"):

    if not os.path.isdir(direc):
        direc = os.path.join(os.getcwd(), direc)
    files = os.listdir(direc)
    files = [os.path.join(direc, file) for file in files]
    for file in files:
        print('checking file: {}'.format(file))
        if file.endswith('.txt'):
            with open(file, 'r') as f:
                try:
                    t = float(f.read().strip())
                except:
                    t = 100
                print(t)
            if t < 100 and file.replace('.txt', suffix) in files and not file.replace('.txt', '.finished') in files:
                print('Deleting file: {}'.format(file))
                os.remove(file)
                os.remove(file.replace('.txt', suffix))


def trim_image(images, player=1, prefix='trimmed_', overwrite=True, remove_timer=True, save_file=False):
    """

    :param images:
    :param player:
    :param prefix:
    :param overwrite:
    :param remove_timer:
    :param save_file:
    :return:
    """
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


def identical_images(image1, image2):
    """

    :param image1:
    :param image2:
    :return:
    """
    img1 = Image.open(image1)
    img1.load()
    img2 = Image.open(image2)
    img2.load()
    data1 = np.asarray(img1)
    data2 = np.asarray(img2)
    return np.array_equal(data1, data2)


def remove_redundant_files(filenames, pressed_buttons):
    """

    :param filenames:
    :param pressed_buttons:
    :return:
    """
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


def timeit(method):
    def timed(*args, **kw):
        t0 = time.time()
        result = method(*args, **kw)
        t1 = time.time()
        print('{}  {} ms'.format(method.__name__, (t1 - t0) * 1000))
        return result
    return timed


def load_model(modelname):
    with open('{}.json'.format(modelname), 'r') as f:
        model = keras.models.model_from_json(f.read())
    model.load_weights('{}_weights.h5'.format(modelname))
    return model