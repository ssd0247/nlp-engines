import time
import logging
import threading
import concurrent.futures as cf

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2) # simulates 2 secs worth of work
    logging.info("Thread %s: finishing", name)

class DatabaseFacade:
    def __init__(self):
        self.value = 0
        # lock as a remedy to the data race conditions
        # with respect to (w.r.t.) the `self.value`
        self._lock = threading.Lock()
        
    def update(self, name):
        logging.info("Thread %s: starting update", name)
        logging.debug("Thread %s about to lock", name)
        with self._lock:
            logging.debug("Thread %s has lock", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.debug("Thread %s about to release lock", name)
        logging.debug("Thread %s after release", name)
        logging.info("Thread %s: finishing update", name)
    
        #local_copy = self.value
        #local_copy += 1
        #time.sleep(0.1)
        #self.value = local_copy
        #logging.info("Thread %s: finishing update", name)

import random

SENTINEL = object()

def producer(pipeline):
    """Pretend we're getting a message from the network."""
    for index in range(10):
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        time.sleep(0.2) # simulate longer(5)/smaller(0.5) network access times!!
        pipeline.set_message(message, "Producer")
    
    # Send a sentinel message to tell consumer we're done
    pipeline.set_message(SENTINEL, "Producer")

def consumer(pipeline):
    """Pretend we're saving a number in the database."""
    message = 0
    while message is not SENTINEL:
        message = pipeline.get_message("Consumer")
        if message is not SENTINEL:
            time.sleep(1) # simulate longer(5)/smaller(0.5) disk writes!!
            logging.info("Consumer storing message: %s", message)

class Pipeline:
    """Class to allow a single element pipeline between producer & consumer."""
    def __init__(self):
        self.message = 0
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()
    
    def get_message(self, name):
        logging.debug("%s:\tabout to acquire lock", name)
        self.consumer_lock.acquire()
        logging.debug("%s:\thave getlock", name)
        message = self.message
        logging.debug("%s:\tabout to release setlock", name)
        self.producer_lock.release()
        logging.debug("%s:\tsetlock released")
        return message
    
    def set_message(self, message, name):
        logging.debug("%s:\tabout to acquire setlock", name)
        self.producer_lock.acquire()
        logging.debug("%s:\thave setlock", name)
        self.message = message
        logging.debug("%s:\tabout to release getlock", name)
        self.consumer_lock.release()
        logging.debug("%s:\tgetlock released", name)

# Solving the above data-race condition using queue instead!!
import queue

def producer_queue(queue, event):
    """Pretend we're getting a number from the network."""
    while not event.is_set():
        message = random.randint(1, 101)
        logging.info("Producer got message: %s", message)
        queue.put(message)
    logging.info("Producer received event. Exiting")

def consumer_queue(queue, event):
    """Pretend we're saving a number in the database."""
    while not event.is_set() and not queue.empty():
        message = queue.get()
        logging.info(
            "Consumer storing message: %s (size=%d)", message, queue.qsize())
    logging.info("Consumer received event. Exiting")



if __name__ == '__main__':
    _format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=_format, level=logging.INFO, datefmt="%H:%M:%S")
    
    threads = list()
    for index in range(1, 4, 1):
        logging.info("Main:\tCreate and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads, start=1):
        logging.info("Main:\tBefore joining thread %d.", index)
        thread.join()
        logging.info("Main:\tThread %d done.", index)
    
    with cf.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function, range(3))
    
    print("*" * 30, "\n", "*" * 30)

    logging.getLogger().setLevel(logging.DEBUG)
    
    database = DatabaseFacade()
    logging.info("Testing update. Starting value is %d.", database.value)
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            executor.submit(database.update, index)
    logging.info("Testing update. Ending value is %d.", database.value)
    
    print("*" * 30, "\n", "*" * 30)

    logging.getLogger().setLevel(logging.INFO)

    pipeline = Pipeline()
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer, pipeline)
        executor.submit(consumer, pipeline)
    
    pipeline_new = queue.Queue()
    event = threading.Event()
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(producer_queue, pipeline_new, event)
        executor.submit(consumer_queue, pipeline_new, event)

        time.sleep(0.1)
        logging.info("Main: about to set event")
        event.set()