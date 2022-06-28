import multiprocessing as mp
import ctypes
import time
import io
import os

from faster_fifo import Queue

import setproctitle


class MpLazyLoadWrapper:
    """
        An multiprocessing object wrapper that allocates (unpickles) the object only when used.
    """

    def __init__(self, wrapped):
        self._packed = None
        self._wrapped = wrapped

    def __getstate__(self):
        buf = io.BytesIO()
        mp.reducer.ForkingPickler(buf).dump(self._wrapped)
        return buf.getvalue()

    def __setstate__(self, state):
        self._packed = state
        self._wrapped = None

    def __getattr__(self, name):
        if self._wrapped is None:
            self._wrapped = mp.reducer.ForkingPickler.loads(self._packed)

        return getattr(self._wrapped, name)


class PickleCheck:
    def __init__(self, shm):
        print("PickleCheck __init__() !")

        if shm:
            print ("Allocate SHM!")
            self.sh_arrays = [mp.RawArray(ctypes.c_uint8, 1048576) for _ in range(256)]  # ~256M shared mem
            # other mp objects
            self.sh_sem = mp.BoundedSemaphore(1)
            self.sh_queue = mp.SimpleQueue()
            self.sh_fifo = Queue(max_size_bytes=1048576)
            self.sh_lock = mp.Lock()
            print (self.__dict__)

    def __getstate__(self):
        print ("PickleCheck Being Pickled !")

        return self.__dict__

    def __setstate__(self, state):
        print ("PickleCheck Unpickled !")

        self.__dict__.update(state)

    def test(self):
        print ("Test")



def worker_(name, obj, test, *args):
    setproctitle.setproctitle(name)

    print ("{}: FDs open".format(name))
    pid = os.getpid()
    os.system("ls -l /proc/%s/fd" %pid)

    if test:
        print ("Calling obj.test()")
        obj.test()

    time.sleep(1000)


if __name__ == "__main__":
    setproctitle.setproctitle("-TestMaster-")
    mp.set_start_method('spawn')

    obj = PickleCheck(shm=True)
    
    # Not lazy
    print ("Starting process that is not lazy...")
    for idx in range(3):
        proc = mp.Process(target=worker_, args=("-Test- {}".format(idx), obj, False))
        proc.start()

        time.sleep(1)

    # Not lazy + access
    print ("Starting process that is not lazy and accesses object...")
    for idx in range(3):
        proc = mp.Process(target=worker_, args=("-TestAccess- {}".format(idx), obj, True))
        proc.start()

        time.sleep(1)

    # Lazy
    print ("Starting process that is lazy...")
    for idx in range(3):
        proc = mp.Process(target=worker_, args=("-LazyTest- {}".format(idx), MpLazyLoadWrapper(obj), False))
        proc.start()

        time.sleep(1)

    # Lazy + Access
    print ("Starting process that is lazy and accesses object...")
    for idx in range(3):
        proc = mp.Process(target=worker_, args=("-LazyAccess- {}".format(idx), MpLazyLoadWrapper(obj), True, MpLazyLoadWrapper(obj)))
        proc.start()

        time.sleep(1)

    print ("Start ok ~")

    # sleep
    time.sleep(1000)
