from __future__ import annotations
import functools
from queue import LifoQueue
import threading


def BufferredEnvWrapper(env, wait=False, buffersize=-1):

    buffer = LifoQueue(buffersize)
    _step = env.step

    @functools.wraps(_step)
    def step(self, action, *a, **kw):
        if wait:
            return _step(action, *a, **kw)
        else:
            threading.Thread(None, lambda: buffer.put(_step(action, *a, **kw)))
            if buffer.qsize() == 0 and not wait:
                return None
            return buffer.get()

    env.buffer = buffer
    env.step = step
    return env
