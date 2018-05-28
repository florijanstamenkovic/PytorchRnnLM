from time import time

class LogTimer:
    """ Utility for emitting logs peridically. """

    def __init__(self, period):
        self._period = period
        self._last_emit = time()

    def __call__(self):
        current = time()
        if current > self._last_emit + self._period:
            self._last_emit = current
            return True
        return False
