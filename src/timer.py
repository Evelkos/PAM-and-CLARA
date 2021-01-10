import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self._is_running = False

    @property
    def time(self):
        """
        Elapsed time in seconds.
        """
        if not self.start_time or not self.end_time or self.start_time > self.end_time:
            raise TimerError(
                "You need to use .start() and .stop() to measure elapsed time."
            )
        return self.end_time - self.start_time

    def start(self):
        self.end_time = None

        if self._is_running:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._is_running = True
        self.start_time = time.perf_counter()

    def stop(self):
        if not self._is_running:
            raise TimerError("Timer is not running. Use .start() to start it")

        self._is_running = False
        self.end_time = time.perf_counter()
