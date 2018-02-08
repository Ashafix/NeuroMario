import os
from PIL import Image


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
