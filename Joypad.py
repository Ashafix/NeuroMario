import numpy as np


class Joypad:
    """
    LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|#P2 Up|P2 Down|P2 Left|P2 Right|P2 Select|P2 Start|P2 Y|P2 B|P2 X|P2 A|P2 L|P2 R|
    """
    empty =   b'|..|............|............|'
    up =      b'|..|U...........|............|'
    down =    b'|..|.D..........|............|'
    left =    b'|..|..L.........|............|'
    right =   b'|..|...R........|............|'
    select =  b'|..|....s.......|............|'
    start =   b'|..|.....S......|............|'
    Y =       b'|..|......Y.....|............|'
    B =       b'|..|.......B....|............|'
    X =       b'|..|........X...|............|'
    A =       b'|..|.........A..|............|'
    L1 =      b'|..|..........l.|............|'
    R1 =      b'|..|...........r|............|'
    default = b'|..|............|............|'
    all_but = b'|..|UDLRsSYBXAlr|............|'
    l_all = list(all_but)
    l_none = list(empty)
    one_player_empty = b'............'
    one_player_all = b'UDLRsSYBXAlr'

    def __init__(self):
        return

    def __repr__(self):
        for att in dir(self):
            if not att.startswith('__') and not callable(getattr(self, att)):
                print(getattr(self, att))

    @staticmethod
    def joypad_to_array(joypad):
        """
        Converts a joypad input to an numpy array
        :param joypad:
        :return:
        """

        if isinstance(joypad, bytes):
            return np.array([(j != 46) * 1 for j in joypad], dtype=np.int8)
        else:
            return np.array([(j != '.') * 1 for j in joypad], dtype=np.int8)

    @staticmethod
    def array_to_joypad(array, player=1, threshold=0.4, missing=None, bit_array=False):
        """

        :param array:
        :return:
        """

        if bit_array:
            index_to_joypad = [Joypad.default,
                               Joypad.B,
                               Joypad.merge(Joypad.B, Joypad.right),
                               Joypad.merge(Joypad.B, Joypad.left)
                               ]
            offset = 4 - array.shape[0]
            if offset not in (0, 1):
                raise ValueError('array dimensions must be either [n][3] or [n][4]')
            index = np.where(array == np.max(array))[0][0]
            return index_to_joypad[index + offset]

        if player > 0:
            offset = 4 + (player - 1) * 13
        else:
            offset = 0
        return_bytes = Joypad.l_none.copy()

        # assume input array and output string have the same length
        if missing is None or len(missing) == 0:
            for i, a in enumerate(array):
                if a > threshold:
                    return_bytes[offset + i] = Joypad.all_but[offset + i]
            return bytes(return_bytes)

        # if only part of the output is present in the input
        if len(missing) + len(array) != len(Joypad.one_player_empty):
            print(missing)
            print(array)
            raise ValueError('if `missing` is used, length of input array and missing needs to be equal to output')

        index = 0
        for i in range(offset, offset + len(Joypad.one_player_empty)):
            if i - offset not in missing:
                if array[index] > threshold:
                    return_bytes[i] = Joypad.all_but[i]
                index += 1
        return bytes(return_bytes)

    @staticmethod
    def flip_input(joypad, down_up=True, left_right=True):
        """
        Flips the input, i.e. down becomes up, right becomes left
        :param input:
        :return:
        """


        if isinstance(joypad, np.ndarray):
            # TODO better make separate function to change the array
            joypad = Joypad.array_to_joypad(joypad, player=1)
            output = 'np.ndarray'
        elif isinstance(joypad, bytes):
            output = 'bytes'
        else:
            raise TypeError('input must be either Numpy array or bytes, found: {}'.format(type(joypad)))

        if down_up:
            index = joypad.find(b'U')
            if index != -1:
                while index != -1:
                    joypad = joypad[0:index] + b'.D' + joypad[index + 2:]
                    index = joypad.find(b'U')
            else:
                index = joypad.find(b'D')
                while index != -1:
                    joypad = joypad[0:index - 1] + b'U.' + joypad[index + 1:]
                    index = joypad.find(b'D')

        if left_right:
            index = joypad.find(b'L')
            if index != -1:
                while index != -1:
                    joypad = joypad[0:index] + b'.R' + joypad[index + 2:]
                    index = joypad.find(b'L')
            else:
                index = joypad.find(b'R')
                while index != -1:
                    joypad = joypad[0:index - 1] + b'L.' + joypad[index + 1:]
                    index = joypad.find(b'R')

        if output == 'np.ndarray':
            return Joypad.joypad_to_array(joypad[4:16])
        else:
            return joypad

    def flip_input_bytes(self, joypad, down_up=True, left_right=True):
        """

        :param joypad:
        :param down_up:
        :param left_right:
        :return:
        """
        if down_up:
            index = joypad.find(b'U')
            if index != -1:
                while index != -1:
                    joypad = joypad[0:index] + b'.D' + joypad[index + 2:]
                    index = joypad.find(b'U')
            else:
                index = joypad.find(b'D')
                while index != -1:
                    joypad = joypad[0:index - 1] + b'U.' + joypad[index + 1:]
                    index = joypad.find(b'D')

        if left_right:
            index = joypad.find(b'L')
            if index != -1:
                while index != -1:
                    joypad = joypad[0:index] + b'.R' + joypad[index + 2:]
                    index = joypad.find(b'L')
            else:
                index = joypad.find(b'R')
                while index != -1:
                    joypad = joypad[0:index - 1] + b'L.' + joypad[index + 1:]
                    index = joypad.find(b'R')

        return joypad

    def flip_input_array(self, joypad, down_up=True, left_right=True, threshold=0.0):
        # TODO unittests
        if not isinstance(joypad, np.ndarray):
            raise TypeError('input must be Numpy array, found: {}'.format(type(joypad)))
        if joypad.shape != (12,) or joypad.shape != (30,):
            raise ValueError('invalid shape, shape must be (12,) or (30,), found {}'.format(joypad.shape))

        if joypad.shape == (12,):
            players = 1
        else:
            players = 2

        offset = 4
        while players > 0:
            if down_up:
                if joypad[offset] > threshold and joypad[offset + 1] <= threshold:
                    joypad[offset] = 0
                    joypad[offset + 1] = 1
                elif joypad[offset + 1] > threshold and joypad[offset] <= threshold:
                    joypad[offset] = 1
                    joypad[offset + 1] = 0
            if left_right:
                if joypad[offset + 2] > threshold and joypad[offset + 3] <= threshold:
                    joypad[offset + 2] = 0
                    joypad[offset + 3] = 1
                elif joypad[offset + 3] > threshold and joypad[offset + 2] <= threshold:
                    joypad[offset + 3] = 0
                    joypad[offset + 2] = 1
            players -= 1
            offset += 13
        return joypad

    @staticmethod
    def merge(key1, key2):
        """

        :param key1:
        :param key2:
        :return:
        """

        return_buttons = Joypad.default
        new_return_buttons = b''
        for i, k in enumerate(key1):
            if k != b'.'[0]:
                new_return_buttons = new_return_buttons + key1[i:i + 1]
            elif key2[i] != b'.'[0]:
                new_return_buttons = new_return_buttons + key2[i:i + 1]
            else:
                new_return_buttons += b'.'

            if i < len(key1) - 1:
                return_bytes = new_return_buttons + return_buttons[i + 1:]
            else:
                return_bytes = new_return_buttons
        return return_bytes


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
