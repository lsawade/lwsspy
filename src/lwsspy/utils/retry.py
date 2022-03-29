import socket
import typing
import time


class retry():
    """
    Decorator that will keep retrying the operation after a timeout.

    Useful for remote operations that are prone to fail spuriously.
    """
    def __init__(self, retries: int, wait_time: float):
        self.retries = retries
        self.wait_time = wait_time

    def __call__(self, f: typing.Callable):
        def wrapped_f(*args, **kwargs):
            for _ in range(self.retries):
                try:
                    retval = f(*args, **kwargs)
                except socket.timeout:
                    time.sleep(self.wait_time)
                    continue
                else:
                    return retval
                raise

        return wrapped_f