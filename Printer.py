class Printer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def log(self, msg, verbose=None):
        if (verbose is True) or (self.verbose and verbose is not False):
            print(msg)
