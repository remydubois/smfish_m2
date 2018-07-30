from .image import *
from os import listdir
import threading
import time
import re
import queue
import multiprocessing
import sys
import math


def wait_for_all(*args):
    while True:
        if not any([a.is_alive() for a in args]):
            time.sleep(0.5)  # Let the time for other threads to finish their job (printing for instance).
            break

        time.sleep(0.05)


def build_path_queue(idir, regex='.'):
    files = [f for f in listdir(idir)]
    images = filter(lambda s: s.lower().endswith('.tif') and re.search(regex, s), files)

    paths = map(lambda i: idir + i, images)

    path_queue = multiprocessing.JoinableQueue()
    steps = 0
    for p in paths:
        path_queue.put(p)
        steps += 1

    # Let it constitute the queue. Just for safety.
    time.sleep(1)

    return path_queue, steps


def generator_queue(path_queue, max_q_size=2,
                    wait_time=0.05):
    the_queue = queue.Queue(max_q_size)
    # This one is used to prevent processing exceptions. I could just use "while not path_queue.Empty()" if I didnt care
    _stop = multiprocessing.Event()

    def data_generator_task():
        while not _stop.is_set() and not path_queue.empty():
            try:
                if not the_queue.full():
                    try:
                        f_p = path_queue.get(False)
                        fish = FQimage(verbose=0)
                        fish.load(f_p)
                        the_queue.put(fish)
                        path_queue.task_done()

                    except queue.Empty:
                        _stop.set()
                        break
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                raise

    feed_thread = threading.Thread(target=data_generator_task)
    feed_thread.daemon = True
    feed_thread.start()

    return the_queue, _stop


def adapt(func):
    """
    This function takes a random function and turn it 'parallelizable'. In more details, it enforce this function
    to expect input from local queue, itself fed from a global queue which feeds all the processes.
    :param func: any function that operate on FQimage instances and return a STRING of information about this image.
    :return: a func performing the same operation but expecting its input from a reachable queue.
    """

    def task(path_q, output_q, **kwargs):
        q, _stop = generator_queue(path_q, max_q_size=2)
        _local_stop = threading.Event()

        def process():
            while not _local_stop.is_set():
                try:
                    im = q.get(False)
                    out = func(im, **kwargs)
                    # print(out)
                    output_q.put(out)
                    q.task_done()
                except queue.Empty:
                    time.sleep(0.1)

        consumer = threading.Thread(target=process)
        # print(multiprocessing.current_process().name, 'started.')
        consumer.start()

        # This one waits for the global queue to be unloaded
        path_q.join()
        # This one wait for the local queue to be emptied.
        q.join()
        # Now I can stop local operations
        _local_stop.set()
        consumer.join()
        # print(multiprocessing.current_process().name, 'stopped.')

    return task


def pump_results(out, output_q, steps):
    sys.stdout.write('\r')
    sys.stdout.write("|%-30s| %.0f%%" % (u"\u2588" * 0, round(1 / 30 * 0, 2)))
    sys.stdout.flush()
    while not len(out) == steps:
        try:
            o = output_q.get(False)
            out.append(o)
            sys.stdout.write('\r')
            sys.stdout.write("|%-30s| %.0f%% " % (
                u"\u2588" * math.floor(len(out) * 30 / steps), round(len(out) * 100 / steps, 2)))
            sys.stdout.flush()
            output_q.task_done()
        except queue.Empty:
            time.sleep(0.01)
