"""Atomic persistence, exclusive run locks and recoverable random state."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import random
import signal
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@contextlib.contextmanager
def atomic_target(path):
    """Commit only fully written files, on the same filesystem as their target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_json(value, path):
    with atomic_target(path) as temporary:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def save_numpy(value, path):
    with atomic_target(path) as temporary:
        with temporary.open('wb') as stream:
            np.save(stream, value, allow_pickle=False)


def save_torch(value, path):
    with atomic_target(path) as temporary:
        torch.save(value, temporary)


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for section, key in [('data', 'root'), ('data', 'cache_directory'), ('output', 'directory')]:
        value = Path(config[section][key]).expanduser()
        config[section][key] = str(value if value.is_absolute() else path.parent / value)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def select_device(requested):
    device = torch.device('cuda' if requested == 'auto' and torch.cuda.is_available()
                          else 'cpu' if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    return device


@contextlib.contextmanager
def exclusive_lock(directory):
    """POSIX advisory lock, released automatically on process termination."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process holds the lock: {directory}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class StopRequest:
    """Signal handler only sets a flag; persistence occurs at a safe boundary."""
    def __init__(self):
        self.requested = False
        self.reason = None
        self.previous = {}

    def __enter__(self):
        def handler(number, frame):
            self.requested = True
            self.reason = signal.Signals(number).name
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            self.previous[sig] = signal.signal(sig, handler)
        return self

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()

