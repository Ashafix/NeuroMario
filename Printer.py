import sys


class Printer:
    def __init__(self, verbose=False, target=sys.stdout):
        self.verbose = verbose
        self.target = target

    def log(self, msg, verbose=None):
        if verbose or (self.verbose and not verbose):
            if self.target is not None:
                msg = str(msg)
                self.target.write(msg)
                self.target.write('\n')
                self.target.flush()
            else:
                self.print(msg)


class PrinterDummy(Printer):
    def __init__(self, verbose=False, target=None):
        self.verbose = verbose
        self.target = target

    def log(self, msg, verbose=None):
        return None
