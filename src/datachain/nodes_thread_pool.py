import concurrent.futures
import threading
from abc import ABC, abstractmethod


class NodeChunk:
    def __init__(
        self, cache, storage, nodes, size_limit=10 * 1024 * 1024, file_limit=100
    ):
        self.cache = cache
        self.storage = storage
        self.nodes = nodes
        self.size_limit = size_limit
        self.file_limit = file_limit

    def __iter__(self):
        return self

    def next_downloadable(self):
        node = next(self.nodes, None)
        while node and (
            not node.is_downloadable or self.cache.contains(node.to_file(self.storage))
        ):
            node = next(self.nodes, None)
        return node

    def __next__(self):
        node = self.next_downloadable()

        total_size = 0
        total_files = 0
        bucket = []

        while (
            node
            and total_size + node.size < self.size_limit
            and total_files + 1 < self.file_limit
        ):
            bucket.append(node)
            total_size += node.size
            total_files += 1
            node = self.next_downloadable()

        if node:
            bucket.append(node)
            total_size += node.size
            total_files += 1

        if bucket:
            return bucket
        raise StopIteration


class NodesThreadPool(ABC):
    def __init__(self, max_threads):
        self._max_threads = max_threads
        self._thread_counter = 0
        self._thread_lock = threading.Lock()
        self.tasks = set()
        self.canceled = False
        self.th_pool = None

    def run(
        self,
        chunk_gen,
        progress_bar=None,
    ):
        results = []
        self.th_pool = concurrent.futures.ThreadPoolExecutor(self._max_threads)
        try:
            self._thread_counter = 0
            for chunk in chunk_gen:
                if self.canceled:
                    break
                while len(self.tasks) >= self._max_threads:
                    done, _ = concurrent.futures.wait(
                        self.tasks, timeout=1, return_when="FIRST_COMPLETED"
                    )
                    self.done_task(done)

                    self.tasks = self.tasks - done
                    self.update_progress_bar(progress_bar)

                self.tasks.add(self.th_pool.submit(self.do_task, chunk))
                self.update_progress_bar(progress_bar)

            while self.tasks:
                if self.canceled:
                    break
                done, _ = concurrent.futures.wait(
                    self.tasks, timeout=1, return_when="FIRST_COMPLETED"
                )
                task_results = self.done_task(done)
                if task_results:
                    results.extend(task_results)

                self.tasks = self.tasks - done
                self.update_progress_bar(progress_bar)
        except:
            self.cancel_all()
            raise
        else:
            self.th_pool.shutdown()

        return results

    def cancel_all(self):
        self.cancel = True
        # Canceling tasks just in case any of them is scheduled to run.
        # Note that running tasks cannot be canceled, instead we will wait for
        # them to finish when shutting down thread loop executor by calling
        # shutdown() method.
        for task in self.tasks:
            task.cancel()
        if self.th_pool:
            self.th_pool.shutdown()  # this will wait for running tasks to finish

    def update_progress_bar(self, progress_bar):
        if progress_bar is not None:
            with self._thread_lock:
                if self._thread_counter:
                    progress_bar.update(self._thread_counter)
                    self._thread_counter = 0

    def increase_counter(self, value):
        with self._thread_lock:
            self._thread_counter += value

    @abstractmethod
    def do_task(self, chunk):
        pass

    @abstractmethod
    def done_task(self, done):
        pass
