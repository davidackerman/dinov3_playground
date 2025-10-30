from queue import Queue, Empty
from threading import Thread, Event
import torch


class ThreadedPrefetcher:
    def __init__(self, loader, num_prefetch=3):
        self.loader = loader
        self.num_prefetch = num_prefetch

    def __iter__(self):
        queue = Queue(maxsize=self.num_prefetch)
        stop_event = Event()
        exception_holder = [None]  # Store exceptions

        def load_batches():
            try:
                for batch in self.loader:
                    if stop_event.is_set():
                        break
                    queue.put(batch)
            except Exception as e:
                exception_holder[0] = e
            finally:
                # Use try-except to handle case where queue is deleted
                try:
                    queue.put(None)
                except:
                    pass

        thread = Thread(target=load_batches, daemon=True)
        thread.start()

        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                if exception_holder[0]:
                    raise exception_holder[0]
                yield batch

                # CRITICAL: Delete batch after yielding to free memory
                del batch

        finally:
            stop_event.set()

            # Drain and delete queued batches
            try:
                while True:
                    batch = queue.get_nowait()
                    del batch
            except Empty:
                pass

            thread.join(timeout=2)

    def __len__(self):
        return len(self.loader)
