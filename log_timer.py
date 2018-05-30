from time import time

class LogTimer:
    """ Utility for periodically emitting.
    Usage:
        lt = LogTimer(2)
        while True:
            if lt():
                log("This is logged every 2 sec")
    """

    def __init__(self, period):
        self._period = period
        self._last_emit = time()

    def __call__(self):
        current = time()
        if current > self._last_emit + self._period:
            self._last_emit = current
            return True
        return False
