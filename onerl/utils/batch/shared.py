import multiprocessing as mp

from onerl.utils.shared_array import SharedArray


class BatchShared:
    def __init__(self, shape_dtype, init_ready=True):
        self.data = {k: SharedArray(*v) for k, v in shape_dtype.items()}
        self.ready = mp.BoundedSemaphore(1)
        if not init_ready:
            self.ready.acquire()

    def get(self):
        obj = object()
        obj.__dict__.update({k: v.get() for k, v in self.data.items()})
        return obj

    def set_ready(self):
        self.ready.release()

    def wait_ready(self):
        self.ready.acquire()