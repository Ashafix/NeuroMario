import os
import zipfile


class MovieFile:
    """

    """
    def __init__(self, filename=''):
        self.filename = filename
        self.name = os.path.split(filename)[-1]
        if filename != '' and not os.path.isfile(filename):
            new_filename = os.path.join(os.getcwd(), filename)
            if os.path.isfile(new_filename):
                self.filename = new_filename
            else:
                new_filename = os.path.join(os.getcwd(), 'movies', filename)
            if os.path.isfile(new_filename):
                self.filename = new_filename
            else:
                raise IOError('File could not be found: {}'.format(filename))
        self.pressed_keys = list()
        self.length = -1
        if filename:
            self.parse_movie()

    def __repr__(self):
        return 'Filename: {}\nLength: {}'.format(self.filename,
                                                 self.length)

    def parse_movie(self, log_filename='Input Log.txt'):
        """

        :param log_filename:
        :return:
        """
        if not os.path.isfile(self.filename):
            self.filename = os.path.join('movies_images', self.filename, self.filename)
            if not os.path.isfile(self.filename):
                raise ValueError('Movie file not found in: {}'.format(self.filename))
        zip_file = zipfile.ZipFile(self.filename)
        file_found = -1
        for file_index, filename in enumerate(zip_file.filelist):
            if filename.filename == log_filename:
                file_found = file_index
        if file_found == -1:
            # to do, raise error
            return None
        self.read_log_file(log_filename, file_zip=zip_file)

    @staticmethod
    def parse_log_file(filename_log):
        """

        :param filename_log:
        :return:
        """
        
        #TODO: check if log file is .bk2 or .log
        if isinstance(filename_log, str):
            with open(filename_log, 'rb') as log_file:
                data = log_file.read().strip().splitlines()
        else:
            data = filename_log.read().strip().splitlines()
        return data[2:-1]

    def read_log_file(self, filename_log, file_zip=None):
        """

        :param filename_log:
        :param file_zip:
        :return:
        """
        if file_zip is None:
            log_file = open(filename_log, 'rb')
        else:
            log_file = file_zip.open(filename_log)

        data = self.parse_log_file(log_file)

        self.pressed_keys = data
        self.length = len(self.pressed_keys)
        return self.pressed_keys
