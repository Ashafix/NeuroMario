import psutil
import os
from Printer import Printer
from Joypad import Joypad
import subprocess
import time
import shutil
import zipfile
from MovieFile import MovieFile



class Emuhawk:
    def __init__(self, emuhawk_exe=None,
                 base_dir=None,
                 rom_name='SuperMario.sfc',
                 movies=None,
                 max_threads=100,
                 wait_time=5,
                 socket_ip=None,
                 socket_port=None,
                 http_ip=None,
                 http_port=None,
                 url_get=None,
                 url_post=None,
                 lua_script=None):

        if emuhawk_exe is not None:
            self.emuhawk_exe = emuhawk_exe
        else:
            emuhawk_exe = os.getenv('emuhawk_exe')
        if emuhawk_exe is not None:
            self.emuhawk_exe = emuhawk_exe
        else:
            emuhawk_exe = self.find_emuhawk_exe()
        if emuhawk_exe is None:
            raise ValueError
        else:
            self.emuhawk_exe = emuhawk_exe

        if base_dir is None:
            self.base_dir = os.getcwd()
        else:
            self.base_dir = base_dir
        self.params_png = ' --movie="{}" --dump-type=imagesequence --dump-name="{}" --dump-length={} --dump-close "{}"'
        self.movie = None
        self.movies = dict()
        self.state = None
        self.rom_name = rom_name
        if not os.path.isfile(self.rom_name):
            if os.path.isfile(os.path.join(os.path.dirname(self.emuhawk_exe), self.rom_name)):
                self.rom_name = os.path.join(os.path.dirname(self.emuhawk_exe), self.rom_name)
            else:
                raise ValueError('ROM could not be found')
        self.running_movies = list()
        self.max_threads = max_threads
        self.wait_time = wait_time
        self.printer = Printer(verbose=True)
        self.pid = None
        self.socket_ip = socket_ip
        self.socket_port = socket_port
        self.http_ip = http_ip
        self.http_port = http_port

        if http_ip is not None and http_port is not None:
            self.url_get = 'http://{}:{}/get'.format(http_ip, http_port)
            self.url_post = 'http://{}:{}/post'.format(http_ip, http_port)
        else:
            self.url_get = url_get
            self.url_post = url_post
        self.lua_script = lua_script
        if movies is not None:
            for movie in movies:
                self.append_movie(movie)

    def find_emuhawk_exe(self):
        """
        Tries to locate EmuHawk.exe
        :return: None if not EmuHawk.exe was found, string with location otherwise
        """
        possible_locations = ['../BizHawk/output/EmuHawk.exe']
        for possible_location in possible_locations:
            if possible_location.startswith('..'):
                possible_location = os.path.join(os.getcwd(), possible_location)
            if os.path.isfile(possible_location):
                return possible_location

    def movie_to_png(self, movie, output_filename):
        """
        Converts a movie to a seris of PNG files
        :param movie:
        :param output_filename:
        :return:
        """
        waiting_time = 0
        while len(self.running_movies) >= self.max_threads:
            if not self.remove_finished_jobs():
                time.sleep(self.wait_time)
            waiting_time += self.wait_time
            if waiting_time > 300:
                break

        cmd_call = self.emuhawk_exe + self.params_png.format(movie.filename, output_filename, movie.length,
                                                             self.rom_name)
        self.printer.log(cmd_call)
        new_filename = os.path.join(os.path.split(output_filename)[0], os.path.split(movie.filename)[1])
        try:
            shutil.copy(movie.filename, new_filename)
        except shutil.SameFileError:
            pass
        p = subprocess.Popen(cmd_call)
        self.running_movies.append(dict(name=movie.name,
                                        filename=output_filename,
                                        length=movie.length,
                                        process=p)
                                   )
        zip_file = zipfile.ZipFile(new_filename)
        zip_file.extract('Input Log.txt', os.path.split(new_filename)[0])

    def movies_to_png(self, movies=list(), folders=list()):
        """
        Converts a list of movies to PNG files
        :param movies:
        :param folders:
        :return:
        """
        mismatch = 0
        if len(movies) == 0:
            movies = self.movies
        if len(folders) == 0:
            folders.append(os.path.join(self.base_dir, 'movies_images'))
            mismatch = 1
            for movie in movies.values():
                movie_name = os.path.split(movie.filename)[-1]
                folders.append(os.path.join(self.base_dir, 'movies_images', movie_name))
        #if len(folders) - len(movies) > mismatch:
        #    self.printer.log('movies and folders do not match')
        #    # TO DO, raise error
        #    return None

        for folder in folders:
            if not os.path.isdir(folder):
                moved = False
                if os.path.isfile(folder):
                    shutil.move(folder, folder + "_")
                    moved = True
                os.mkdir(folder)
                if moved:
                    shutil.move(folder + "_", folder)

        for movie in movies.values():
            output_filename = '{}\\movies_images\\{}\\{}_frame.png'.format(self.base_dir, movie.name, movie.name)
            self.movie_to_png(movie, output_filename)

    def append_movie(self, movie):
        """
        Appends a movie to the movie list
        :param movie: string, location of the .bk2 file
        :return:
        """
        if movie in self.movies:
            return False
        self.movies[movie] = MovieFile(filename=movie)
        return True

    def append_movies(self, movies):
        """
        Appends a list of movies to the movie list
        :param movies:
        :return:
        """
        for movie in movies:
            self.append_movie(movie)

    def remove_finished_jobs(self):
        to_be_removed = list()
        for job_info in self.running_movies:
            filename = job_info['name'][0:job_info['name'].rfind('.')]  # prefix
            filename += str(job_info['length'])  # the length of the movie
            filename += job_info['name'][job_info['name'].rfind('.'):]  # suffix
            if os.path.isfile(filename):
                to_be_removed.append(self.running_movies.index(job_info[0]))
        for remove_me in to_be_removed:
            self.running_movies.pop(remove_me)
        return len(self.running_movies) == 0

    def wait_until_finished(self, timeout=300):
        while len(self.running_movies) > 0 and timeout > 0:
            self.remove_finished_jobs()
            if len(self.running_movies) > 0:
                time.sleep(self.wait_time)
                timeout += -self.wait_time
        return timeout >= 0

    def create_defined_state(self, gametype='1P', racetype='Time Trial', player='Mario', cc_class=None,
                             cup='MUSHROOM CUP', track='GHOST VALLEY 1',
                             filename_movie=None, auto_start=True):
        """
        Creates and saves a defined state at the beginning of a race
        :return:
        """

        gametype = gametype.upper()
        racetype = racetype.upper()
        player = player.upper()
        cup = cup.upper()
        track = track.upper()

        if gametype not in ('1P', '2P'):
            raise ValueError
        if racetype not in ('TIME TRIAL', 'MARIOKART GP'):
            raise ValueError
        if cc_class is not None and int(cc_class) not in (50, 100, 150):
            raise ValueError

        valid_players = ('MARIO', 'PRINCESS', 'BOWSER', 'KOOPA', 'LUIGI', 'TOAD', 'DONKEY', 'NO IDEA')
        if player not in valid_players:
            raise ValueError
        valid_cups = ('MUSHROOM CUP', 'FLOWER CUP', 'STAR CUP')

        if cup not in valid_cups:
            raise ValueError
        valid_tracks = ['MARIO CIRCUIT 1', 'DONUT PLAINS 1', 'GHOST VALLEY 1', 'BOWSER CASTLE 1', 'MARIO CIRCUIT 2', 'CHOCO ISLAND 1', 'GHOST VALLEY 2', 'DONUT PLAINS 2', 'BOWSER CASTLE 2', 'MARIO CIRCUIT 3', 'KOOPA BEACH 1', 'CHOCO ISLAND 2', 'VANILLA LAKE 1', 'BOWSER CASTLE 3', 'MARIO CIRCUI 4']
        if track is not None and track not in valid_tracks:
            raise ValueError

        if filename_movie is None:
            filename_movie = os.path.join(os.getcwd(), 'DefinedState.bk2')
        if filename_movie.endswith('.bk2'):
            filename_state = filename_movie[0:-4] + '.state'
        else:
            filename_state = filename_movie + '.state'


        # create "Input log.txt" file
        with open('emptyMovie/Input Log.txt', 'r') as f:
            empty_log = f.readlines()
        output_lines = list()
        output_lines.append(empty_log[0].strip())
        output_lines.append(empty_log[1].strip())

        joypad_down = Joypad.down
        joypad_left = Joypad.left
        joypad_right = Joypad.right
        joypad_select = Joypad.select
        joypad_default = Joypad.empty
        joypad_start = Joypad.start
        # select 1P or 2P
        output_lines.extend([joypad_default] * 262)
        output_lines.extend([joypad_start] * 8)
        output_lines.extend([joypad_default] * 30)
        if gametype == '2P':
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 30)

        output_lines.extend([joypad_start] * 8)
        output_lines.extend([joypad_default] * 30)

        # select Time Trial or Mariokart GP
        if racetype == 'TIME TRIAL':
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 60)
            output_lines.extend([joypad_start] * 8)
            output_lines.extend([joypad_default] * 60)
        else:
            output_lines.extend([joypad_start] * 8)
            output_lines.extend([joypad_default] * 60)
            # select correct class (difficulty)
            while cc_class > 50:
                output_lines.extend([joypad_down] * 8)
                output_lines.extend([joypad_default] * 60)
                cc_class -= 50
            output_lines.extend([joypad_start] * 8)
            output_lines.extend([joypad_default] * 120)

        # press YES
        output_lines.extend([joypad_start] * 8)
        output_lines.extend([joypad_default] * 120)

        # select player
        p = valid_players.index(player)
        if p > 3:
            output_lines.extend([joypad_down] * 8)
            output_lines.extend([joypad_default] * 8)
        while p % 4 > 0:
            output_lines.extend([joypad_right] * 8)
            output_lines.extend([joypad_default] * 8)
            p -= 1

        # press OK
        output_lines.extend([joypad_start] * 8)
        output_lines.extend([joypad_default] * 31)
        output_lines.extend([joypad_start] * 8)
        output_lines.extend([joypad_default] * 240)

        c = valid_cups.index(cup)
        if racetype == 'TIME TRIAL':
            if c > 0:
                output_lines.extend([joypad_left] * 8)
                output_lines.extend([joypad_default] * 8)
            while c > 0:
                output_lines.extend([joypad_down] * 8)
                output_lines.extend([joypad_default] * 8)

            output_lines.extend([joypad_right] * 8)
            output_lines.extend([joypad_default] * 30)

            # select track
            t = valid_tracks.index(track) % 5
            while t > 0:
                output_lines.extend([joypad_down] * 8)
                output_lines.extend([joypad_default] * 8)
                t -= 1

            output_lines.extend([joypad_start] * 8)
            output_lines.extend([joypad_default] * 60)
            output_lines.extend([joypad_start] * 8)

        else:
            while c > 0:
                output_lines.extend([joypad_down] * 8)
                output_lines.extend([joypad_default] * 8)
            output_lines.extend([joypad_start] * 8)
        if racetype == 'TIME TRIAL':
            output_lines.extend([joypad_default] * 150)
        else:
            output_lines.extend([joypad_default] * 450)
        output_lines.append(empty_log[2].strip())

        with open(os.path.join(os.getcwd(), 'emptyMovie', 'Input Log.txt_'), 'w') as f:
            for o in output_lines:
                if isinstance(o, str):
                    f.write(o)
                else:
                    f.write(o.decode())
                f.write('\n')
        zf = zipfile.ZipFile(filename_movie, 'w', zipfile.ZIP_DEFLATED)

        zf.write('emptyMovie/SyncSettings.json', 'SyncSettings.json')
        zf.write('emptyMovie/Subtitles.txt', 'Subtitles.txt')
        zf.write('emptyMovie/Input Log.txt_', 'Input Log.txt')
        zf.write('emptyMovie/Comments.txt', 'Comments.txt')
        zf.write('emptyMovie/Header.txt', 'Header.txt')
        zf.close()

        if auto_start:
            lua_string = """
index = 0
while index < {} do
    emu.frameadvance()
    index = index + 1
end
savestate.save("{}")
""".format(len(output_lines) - 3, filename_state)
            filename_lua = os.path.join(os.getcwd(), 'emptyMovie', 'defined_state.lua')
            with open(filename_lua, 'w') as f:
                f.write(lua_string)
            self.lua_script = filename_lua
        self.movie = filename_movie
        self.start()
        return filename_state

    def start(self):

        args = [self.emuhawk_exe]
        if self.socket_ip is not None or self.socket_port is not None:
            if self.socket_ip is None or self.socket_port is None:
               raise ValueError
            args.append('--socket_ip=')
            args[-1] += self.socket_ip
            args.append('--socket_port=')
            args[-1] += str(self.socket_port)

        if self.url_get is not None or self.url_post is not None:
            if self.url_get is None or self.url_post is None:
                raise ValueError
            args.append('--url_get=')
            args[-1] += self.url_get
            args.append('--url_post=')
            args[-1] += self.url_post
        if self.lua_script is not None:
            args.append('--lua=')
            args[-1] += self.lua_script
        if self.movie is not None:
            args.append('--movie=')
            args[-1] += self.movie
        if self.state is not None:
            args.append('--load-state=')
            args[-1] += self.state

        args.append(self.rom_name)
        print(' '.join(args))
        p = subprocess.Popen(args)
        self.pid = p.pid

    def stop(self):
        if self.pid is None:
            return False

        try:
            subprocess.run('taskkill /F /T /PID {}'.format(self.pid), shell=True)
            return True
        except:
            return False

    def is_running(self):
        return psutil.pid_exists(self.pid)
