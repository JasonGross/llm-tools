import pandas as pd
from functools import wraps
import pickle
from frozendict import frozendict
import threading
from filelock import FileLock
from pathlib import Path
import tempfile, os

__all__ = ['Memoize']

def to_immutable(arg):
    """Converts a list or dict to an immutable version of itself."""
    if isinstance(arg, list):
        return tuple(to_immutable(e) for e in arg)
    elif isinstance(arg, dict):
        return frozendict({k: to_immutable(v) for k, v in arg.items()})
    else:
        return arg

class DummyContextWrapper:
    def __enter__(self): pass
    def __exit__(self, *args): pass

def wrap_context(ctx, skip=False):
    """If skip is True, returns a dummy context manager that does nothing."""
    return ctx if not skip else DummyContextWrapper()

def write_via_temp(file_path, do_write):
    """Writes to a file by writing to a temporary file and then renaming it.
    This ensures that the file is never in an inconsistent state."""
    temp_dir = Path(file_path).parent
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, mode="wb") as temp_file:
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

class Memoize:
    """A memoization decorator that caches the results of a function call.
    The cache is stored in a file, so it persists between runs.
    The cache is also thread-safe, so it can be used in a multithreaded environment.
    The cache is also exception-safe, so it won't be corrupted if there's an error.
    """
    instances = {}

    cache_base_dir = "cache"

    def __init__(self, func, name=None, cache_file=None):
        self.func = func
        self.name = name or func.__name__
        self.cache_file = Path(cache_file or (Path(Memoize.cache_base_dir) / f"{self.name}_cache.pkl"))
        self.cache = {}
        self.df_cache = set()
        self.df = pd.DataFrame(columns=['input', 'output'])
        self.df_thread_lock = threading.Lock()
        self.thread_lock = threading.Lock()
        self.file_lock = FileLock(f"{self.cache_file}.lock")
        Memoize.instances[self.name] = self
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache_from_disk()
        for attr in ('__doc__', '__name__', '__module__'):
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

    def _load_cache_from_disk(self, use_lock=True):
        """Loads the cache from disk.  If use_lock is True, then the cache is locked while it's being loaded."""
        with wrap_context(self.file_lock, skip=not use_lock):
            try:
                with open(self.cache_file, "rb") as f:
                    disk_cache = pickle.load(f)
            except FileNotFoundError:
                return
        with wrap_context(self.thread_lock, skip=not use_lock):
            self.cache.update(disk_cache)

    def _write_cache_to_disk(self):
        """Writes the cache to disk.  The cache is locked while it's being written."""
        with self.thread_lock:
            with self.file_lock:
                self._load_cache_from_disk(use_lock=False)
                # use a tempfile so that we don't corrupt the cache if there's an error
                write_via_temp(self.cache_file, (lambda f: pickle.dump(self.cache, f)))

    @classmethod
    def sync_all(cls):
        """Writes all caches to disk."""
        for instance in cls.instances.values():
            instance._write_cache_to_disk()

    def __call__(self, *args, **kwargs):
        """Calls the function, caching the result if it hasn't been called with the same arguments before."""
        immutable_args = to_immutable(args)
        immutable_kwargs = to_immutable(kwargs)
        key = (immutable_args, immutable_kwargs)

        self._load_cache_from_disk()

        if key not in self.cache.keys():
            val = self.func(*args, **kwargs)
            with self.thread_lock: self.cache[key] = val
            self._write_cache_to_disk()
        else:
            with self.thread_lock: val = self.cache[key]
        
        with self.df_thread_lock:
            if key not in self.df_cache:
                self.df_cache.add(key)
                new_row = pd.DataFrame({'input': [key], 'output': [val]})
                self.df = pd.concat([self.df, new_row], ignore_index=True)
        return val
    
    def __repr__(self):
        return f"Memoize(func={self.func!r}, name={self.name!r}, cache_file={self.cache_file!r})"
    
    def __str__(self):
        return f"Memoize(func={self.func}, name={self.name}, cache_file={self.cache_file})"
