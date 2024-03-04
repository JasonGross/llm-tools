import inspect
import pickle
from typing import IO, Any, Callable, Optional, Tuple, TypeVar, Union
from contextlib import (
    AbstractContextManager,
    contextmanager,
    asynccontextmanager,
)
import concurrent.futures
from frozendict import frozendict
import threading
from filelock import FileLock
from pathlib import Path
import asyncio
import tempfile, os

pd = None
try:
    if not os.getenv("MEMOCACHE_NO_PANDAS"):
        import pandas as pd
except ImportError:
    pass

__all__ = ["Memoize", "USE_PANDAS"]

USE_PANDAS = pd is not None

T = TypeVar("T", bound=AbstractContextManager)
KEY = Tuple[tuple, frozendict]


def to_immutable(arg: Any) -> Any:
    """Converts a list or dict to an immutable version of itself."""
    if isinstance(arg, list) or isinstance(arg, tuple):
        return tuple(to_immutable(e) for e in arg)
    elif isinstance(arg, dict):
        return frozendict({k: to_immutable(v) for k, v in arg.items()})
    else:
        return arg


class DummyContextWrapper(AbstractContextManager):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def wrap_context(ctx: T, skip: bool = False) -> Union[T, DummyContextWrapper]:
    """If skip is True, returns a dummy context manager that does nothing."""
    return ctx if not skip else DummyContextWrapper()


def write_via_temp(file_path: Union[str, Path], do_write: Callable[[IO[bytes]], Any]):
    """Writes to a file by writing to a temporary file and then renaming it.
    This ensures that the file is never in an inconsistent state."""
    temp_dir = Path(file_path).parent
    with tempfile.NamedTemporaryFile(
        dir=temp_dir, delete=False, mode="wb"
    ) as temp_file:
        temp_file_path = temp_file.name
        try:
            do_write(temp_file)
        except Exception:
            # If there's an error, clean up the temporary file and re-raise the exception
            temp_file.close()
            os.remove(temp_file_path)
            raise
    # Rename the existing cache file to a backup file
    backup_file_path = f"{file_path}.bak"
    try:
        if os.path.exists(file_path):
            os.rename(file_path, backup_file_path)

        # Rename the temporary file to the cache file
        os.rename(temp_file_path, file_path)
    finally:
        # Delete the backup file
        if os.path.exists(backup_file_path):
            if not os.path.exists(file_path):
                os.rename(backup_file_path, file_path)
            else:
                os.remove(backup_file_path)


# https://stackoverflow.com/a/63425191/377022
_pool = concurrent.futures.ThreadPoolExecutor()


@asynccontextmanager
async def async_lock(lock):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_pool, lock.acquire)
    try:
        yield  # the lock is held
    finally:
        lock.release()


class Memoize:
    """A memoization decorator that caches the results of a function call.
    The cache is stored in a file, so it persists between runs.
    The cache is also thread-safe, so it can be used in a multithreaded environment.
    The cache is also exception-safe, so it won't be corrupted if there's an error.
    """

    instances = {}

    cache_base_dir = "cache"

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        cache_file: Optional[Union[str, Path]] = None,
        disk_write_only: bool = False,
        use_pandas: Optional[bool] = None,
        force_async: bool = False,
        process_half_async: bool = True,
    ):
        """Initializes the memoization decorator.

        process_half_async: if a synchronous function returns a tuple or list of coroutines, then, when this is true, we jump through some extra hoops to ensure that the coroutines are awaited before the result is cached.
        """
        if use_pandas is None:
            use_pandas = USE_PANDAS
        if isinstance(func, Memoize):
            self.func = func.func
            self.name: str = name or func.name
            self.cache_file = Path(cache_file or func.cache_file).absolute()
            self.cache: dict = func.cache
            self.df_cache: set = func.df_cache
            self.df = func.df
            self.df_thread_lock: threading.Lock = func.df_thread_lock
            self.thread_lock: threading.Lock = func.thread_lock
            self.file_lock = func.file_lock
            self.force_async = func.force_async
            self.process_half_async = func.process_half_async
            if name is not None:
                Memoize.instances[name] = self
        else:
            self.func = func
            self.name = name or func.__name__
            self.cache_file = Path(
                cache_file or (Path(Memoize.cache_base_dir) / f"{self.name}_cache.pkl")
            ).absolute()
            self.cache: dict = {}
            self.df_cache: set = set()
            self.df = (
                pd.DataFrame(columns=["input", "output"])
                if use_pandas and pd is not None
                else None
            )
            self.df_thread_lock: threading.Lock = threading.Lock()
            self.thread_lock: threading.Lock = threading.Lock()
            self.file_lock: FileLock = FileLock(f"{self.cache_file}.lock")
            self.force_async = force_async
            self.process_half_async = process_half_async
            Memoize.instances[self.name] = self
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_cache_from_disk()
        self.disk_write_only: int = disk_write_only
        self.disk_write_only_lock: threading.Lock = threading.Lock()
        for attr in ("__doc__", "__name__", "__module__"):
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

    def _load_cache_from_disk(self, use_lock: bool = True):
        """Loads the cache from disk.  If use_lock is True, then the cache is locked while it's being loaded."""
        with wrap_context(self.file_lock, skip=not use_lock):
            try:
                with open(self.cache_file, "rb") as f:
                    disk_cache = pickle.load(f)
            except FileNotFoundError:
                return
        with wrap_context(self.thread_lock, skip=not use_lock):
            self.cache.update(disk_cache)

    def _write_cache_to_disk(self, skip_load: bool = False, use_lock: bool = True):
        """Writes the cache to disk.  The cache is locked while it's being written."""
        with wrap_context(self.thread_lock, skip=not use_lock):
            with wrap_context(self.file_lock, skip=not use_lock):
                if not skip_load:
                    self._load_cache_from_disk(use_lock=False)
                # use a tempfile so that we don't corrupt the cache if there's an error
                write_via_temp(self.cache_file, (lambda f: pickle.dump(self.cache, f)))

    def kwargs_of_key(self, key: KEY) -> frozendict:
        """Returns the kwargs of a key."""
        return key[1]

    def args_of_key(self, key: KEY) -> tuple:
        """Returns the args of a key."""
        return key[0]

    def _uncache(self, key: KEY):
        """Removes a key from the cache."""
        with self.thread_lock:
            with self.file_lock:
                self._load_cache_from_disk(use_lock=False)
                del self.cache[key]
                self._write_cache_to_disk(skip_load=True, use_lock=False)

    def uncache(self, *args, **kwargs):
        return self._uncache((to_immutable(args), to_immutable(kwargs)))

    @classmethod
    def sync_all(cls):
        """Writes all caches to disk."""
        for instance in cls.instances.values():
            instance._write_cache_to_disk()

    def key_of_args(self, *args, **kwargs) -> KEY:
        return (to_immutable(args), to_immutable(kwargs))

    def _update_df(self, key: KEY, val: Any):
        if self.df is not None and pd is not None:
            with self.df_thread_lock:
                if key not in self.df_cache:
                    self.df_cache.add(key)
                    new_row = pd.DataFrame({"input": [key], "output": [val]})
                    self.df = pd.concat([self.df, new_row], ignore_index=True)

    def _process_half_async_result(self, key, val):
        if (isinstance(val, tuple) or isinstance(val, list)) and any(
            inspect.isawaitable(x) for x in val
        ):
            list_vals = list(val)

            def process_val(i, v):
                if inspect.isawaitable(v):

                    async def await_v(v):
                        v = await v
                        list_vals[i] = v
                        if not any(inspect.isawaitable(x) for x in list_vals):
                            await self._async_overwrite_result(
                                key, type(val)(list_vals)
                            )
                        return v

                    return await_v(v)
                return v

            return type(val)(process_val(i, v) for i, v in enumerate(val))
        else:
            self._sync_overwrite_result(key, val)
            return val

    def _sync_overwrite_result(self, key, val):
        with self.thread_lock:
            self.cache[key] = val
        self._write_cache_to_disk()

    async def _async_overwrite_result(self, key, val):
        async with async_lock(self.thread_lock):
            self.cache[key] = val
        await asyncio.to_thread(self._write_cache_to_disk)

    def _sync_call(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        key = self.key_of_args(*args, **kwargs)

        if not self.disk_write_only:
            self._load_cache_from_disk()

        if key not in self.cache.keys():
            val = self.func(*args, **kwargs)
            val = (
                self._sync_overwrite_result(key, val)
                if not self.process_half_async
                else self._process_half_async_result(key, val)
            )
        else:
            with self.thread_lock:
                val = self.cache[key]

        self._update_df(key, val)
        return val

    async def _async_call(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        key = self.key_of_args(*args, **kwargs)

        if not self.disk_write_only:
            await asyncio.to_thread(self._load_cache_from_disk)

        if key not in self.cache.keys():
            val = await self.func(*args, **kwargs)
            await self._async_overwrite_result(key, val)
        else:
            async with async_lock(self.thread_lock):
                val = self.cache[key]

        self._update_df(key, val)
        return val

    def __call__(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        if inspect.iscoroutinefunction(self.func) or self.force_async:
            return self._async_call(*args, **kwargs)
        else:
            return self._sync_call(*args, **kwargs)

    @contextmanager
    def sync_cache(self, inplace: bool = False):
        """Syncs the cache to disk on entering the context.

        If inplace is False, returns a copy of the function that can be called without incurring the overhead of reading from disk.
        If inplace is True, mutates the current function to avoid reading from disk.
        """
        self._load_cache_from_disk()
        if inplace:
            with self.disk_write_only_lock:
                self.disk_write_only += 1
            try:
                yield self
            finally:
                with self.disk_write_only_lock:
                    self.disk_write_only -= 1
        else:
            yield Memoize(self, disk_write_only=True)

    def __repr__(self):
        return f"Memoize(func={self.func!r}, name={self.name!r}, cache_file={self.cache_file!r})"

    def __str__(self):
        return (
            f"Memoize(func={self.func}, name={self.name}, cache_file={self.cache_file})"
        )
