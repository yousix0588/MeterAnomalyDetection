"""Bounded-memory external sorting, regular-grid shards and lazy windows.

CSV and Parquet are streamed. Excel must first be converted to CSV/Parquet.
SQLite staging deduplicates raw timestamps before fixed-duration resampling.
No window crosses a meter or train/validation/calibration/test boundary.
"""
from __future__ import annotations
import json
import logging
import math
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from .utils import (atomic_target, digest, exclusive_lock, file_sha256,
                    save_json, save_numpy)

LOG = logging.getLogger(__name__)
FORMAT_VERSION = 1


def parse_series_id(path):
    return Path(path).stem


def discover_files(root, extensions):
    allowed = {x.lower() for x in extensions}
    return sorted(p for p in Path(root).rglob('*') if p.is_file()
                  and p.suffix.lower() in allowed and not p.name.startswith('.'))


def table_chunks(path, columns, rows):
    if path.suffix.lower() == '.csv':
        yield from pd.read_csv(path, usecols=columns, chunksize=rows)
    elif path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=rows, columns=columns):
            yield batch.to_pandas()
    else:
        raise ValueError(f'{path}: convert Excel to CSV or Parquet before large-data preparation')


def source_inventory(config):
    d = config['data']
    files = discover_files(d['root'], d.get('extensions', ['.csv', '.parquet']))
    groups = {}
    pattern = d.get('series_id_regex')
    for path in files:
        if pattern:
            match = re.search(pattern, str(path.relative_to(d['root'])))
            if not match or 'series_id' not in match.groupdict():
                raise ValueError(f'series_id_regex must match {path} with named group series_id')
            key = match.group('series_id')
        else:
            key = parse_series_id(path)
        groups.setdefault(key, []).append(path)
    if d.get('meter_limit') is not None:
        groups = dict(sorted(groups.items())[:int(d['meter_limit'])])
    if not groups:
        raise ValueError('No input files found')
    inventory = {key: [{'path': str(p.resolve()), 'size': p.stat().st_size,
                       'mtime_ns': p.stat().st_mtime_ns} for p in paths]
                 for key, paths in sorted(groups.items())}
    return groups, inventory


def validate_config(config):
    d, s, t = config['data'], config['splits'], config['training']
    w, stride = int(d['window_size']), int(d['stride'])
    block = int(d.get('score_block_size', stride))
    if not (w >= 2 and 1 <= stride <= w and block == stride and w % stride == 0):
        raise ValueError('Require window_size >= 2, 1 <= stride <= window_size, score_block_size == stride, and window_size divisible by stride')
    if not d['features'] or len(set(d['features'])) != len(d['features']):
        raise ValueError('features must be a nonempty unique ordered list')
    if not (0 <= d['max_missing_ratio'] <= 1) or d['max_interpolation_gap'] < 0:
        raise ValueError('Invalid missing-data settings')
    if int(d['shard_rows']) < max(w, d['max_interpolation_gap'] + 1) or int(d['read_chunk_rows']) < 1:
        raise ValueError('shard_rows must be >= window_size and gap+1; read_chunk_rows must be positive')
    if not 0 < s['validation_ratio'] < 1 or not 0 < s['calibration_ratio'] < 1:
        raise ValueError('validation_ratio and calibration_ratio must be positive')
    if s['validation_ratio'] + s['calibration_ratio'] >= 1:
        raise ValueError('validation_ratio + calibration_ratio must be < 1')
    if int(t['batch_size']) < 1 or int(t['epochs']) < 1 or int(t['patience']) < 1:
        raise ValueError('batch_size, epochs and patience must be positive')
    if not 0 < t['threshold_quantile'] < 1:
        raise ValueError('threshold_quantile must be in (0, 1)')
    freq = pd.to_timedelta(d['frequency']).value
    if freq <= 0:
        raise ValueError('frequency must be a positive fixed duration')
    return int(freq)


def boundary(value, timezone):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
    return timestamp.tz_convert('UTC').value


def _stage_series(paths, db_path, config, stop):
    """Resume at source-file boundaries; each source commits in one transaction."""
    d = config['data']
    f = len(d['features'])
    connection = sqlite3.connect(db_path)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA cache_size=-32768')
    connection.execute('PRAGMA temp_store=FILE')
    connection.execute('CREATE TABLE IF NOT EXISTS done(path TEXT PRIMARY KEY, invalid INTEGER)')
    connection.execute('CREATE TABLE IF NOT EXISTS raw(ts INTEGER PRIMARY KEY, ' +
                       ', '.join(f'f{i} REAL' for i in range(f)) + ')')
    query = 'INSERT OR REPLACE INTO raw VALUES (' + ','.join('?' for _ in range(f + 1)) + ')'
    try:
        for path in paths:
            if connection.execute('SELECT 1 FROM done WHERE path=?', (str(path),)).fetchone():
                continue
            invalid = 0
            with connection:
                for frame in table_chunks(path, [d['timestamp_column'], *d['features']], d['read_chunk_rows']):
                    timestamp = pd.to_datetime(frame[d['timestamp_column']], errors='coerce', utc=True, format='mixed').dt.as_unit('ns')
                    invalid += int(timestamp.isna().sum())
                    frame = frame.loc[timestamp.notna(), d['features']].apply(pd.to_numeric, errors='coerce')
                    values = frame.to_numpy(dtype=np.float64)
                    times = timestamp.loc[timestamp.notna()].astype('int64').to_numpy()
                    # Insert order makes the last occurrence in sorted file order authoritative.
                    connection.executemany(query, ((int(ts), *[float(x) if np.isfinite(x) else None for x in row])
                                                   for ts, row in zip(times, values)))
                    if stop.requested:
                        raise InterruptedError('Preparation interrupted; current source transaction rolled back')
                connection.execute('INSERT INTO done VALUES (?,?)', (str(path), invalid))
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return connection
    except BaseException:
        connection.close()
        raise


def _regular_blocks(connection, start_bin, end_bin, freq, features, rows):
    """Fill absent bins incrementally; query operates on the timestamp index."""
    query = ('SELECT CAST(ts / ? AS INTEGER) AS bucket, ' +
             ','.join(f'AVG(f{i})' for i in range(features)) +
             ' FROM raw WHERE ts >= ? AND ts < ? GROUP BY bucket ORDER BY bucket')
    cursor = iter(connection.execute(query, (freq, start_bin * freq, end_bin * freq)))
    item = next(cursor, None)
    for start in range(start_bin, end_bin, rows):
        size = min(rows, end_bin - start)
        array = np.full((size, features), np.nan, dtype=np.float32)
        while item is not None and item[0] < start + size:
            if item[0] >= start:
                array[item[0] - start] = [np.nan if x is None else x for x in item[1:]]
            item = next(cursor, None)
        yield start, array


def _short_gap_fill(array, gap):
    """Interpolate only entire interior runs <= gap; never extrapolate endpoints."""
    result = array.copy()
    if gap == 0:
        return result
    for feature in range(array.shape[1]):
        x = array[:, feature]
        missing = ~np.isfinite(x)
        edges = np.diff(np.r_[False, missing, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for a, b in zip(starts, ends):
            if b - a <= gap and a > 0 and b < len(x):
                result[a:b, feature] = np.linspace(x[a - 1], x[b], b - a + 2)[1:-1]
    return result


def _write_segment(connection, key, kind, start_bin, end_bin, config, cache, prefix, stop):
    """Own windows by endpoint, retaining left context and interpolation halos."""
    d = config['data']
    rows, w, stride, gap = (int(d[x]) for x in
                            ('shard_rows', 'window_size', 'stride', 'max_interpolation_gap'))
    freq = pd.to_timedelta(d['frequency']).value
    f = len(d['features'])
    # Two small read blocks plus retained context; no complete series resident.
    blocks = iter(_regular_blocks(connection, start_bin, end_bin, freq, f, rows))
    current = next(blocks, None)
    following = next(blocks, None)
    history = np.empty((0, f), np.float32)
    result = []
    shard_number = 0
    while current is not None:
        core_start, core = current
        ahead = following[1][:gap + 1] if following is not None else np.empty((0, f), np.float32)
        combined = np.concatenate([history, core, ahead])
        combined_start = core_start - len(history)
        filled = _short_gap_fill(combined, gap)
        array_start = max(start_bin, core_start - w + 1)
        offset = array_start - combined_start
        size = core_start + len(core) - array_start
        values = filled[offset:offset + size]
        observed = np.isfinite(combined[offset:offset + size]).all(axis=1)
        complete = np.isfinite(values).all(axis=1)
        anchor = start_bin + (stride if kind == 'test' else w) - 1
        first_endpoint = max(core_start, anchor)
        first_endpoint += (-(first_endpoint - anchor)) % stride
        endpoints = np.arange(first_endpoint, core_start + len(core), stride, dtype=np.int64)
        starts = endpoints - w + 1 - array_start
        if len(starts):
            observed_sum = np.r_[0, np.cumsum(observed, dtype=np.int64)]
            complete_sum = np.r_[0, np.cumsum(complete, dtype=np.int64)]
            valid = np.zeros(len(starts), dtype=bool)
            eligible = starts >= 0
            positions = starts[eligible]
            valid[eligible] = ((complete_sum[positions + w] - complete_sum[positions] == w) &
                              ((w - (observed_sum[positions + w] - observed_sum[positions])) <=
                               d['max_missing_ratio'] * w + 1e-9))
        else:
            valid = np.empty(0, bool)
        stem = f'{prefix}_{kind}_{shard_number:06d}'
        paths = {name: f'shards/{stem}.{name}.npy' for name in ('values', 'observed', 'starts', 'valid')}
        for name, array in [('values', values), ('observed', observed),
                            ('starts', starts), ('valid', valid)]:
            save_numpy(array, cache / paths[name])
        record = {'series_id': key, 'split': kind, 'start_bin': int(array_start),
                  'rows': size, 'windows': int(valid.sum()), 'candidates': len(starts),
                  'files': paths, 'sha256': {name: file_sha256(cache / p) for name, p in paths.items()}}
        result.append(record)
        history = np.concatenate([history, core])[-(w - 1 + gap + 1):]
        current, following = following, next(blocks, None)
        shard_number += 1
        if stop.requested:
            raise InterruptedError('Preparation interrupted; incomplete meter will be rebuilt')
    return result


def _sample_scaler(manifest, config, cache, stop):
    """Uniform-priority reservoir over complete training rows, bounded by K."""
    d = config['data']
    capacity = int(d.get('scaler_max_rows', 1_000_000))
    if capacity < 2:
        raise ValueError('scaler_max_rows must be >= 2')
    rng = np.random.default_rng(config['seed'])
    reservoir = np.empty((0, len(d['features'])), np.float32)
    priorities = np.empty(0)
    seen = 0
    last_end = {}
    for shard in manifest['shards']:
        if shard['split'] != 'train':
            continue
        array = np.load(cache / shard['files']['values'], mmap_mode='r')
        # Exclude repeated left context from reservoir sampling.
        offset = max(0, last_end.get(shard['series_id'], shard['start_bin']) - shard['start_bin'])
        last_end[shard['series_id']] = shard['start_bin'] + shard['rows']
        rows = np.asarray(array[offset:])
        rows = rows[np.isfinite(rows).all(axis=1)]
        seen += len(rows)
        new_keys = rng.random(len(rows))
        reservoir = np.concatenate([reservoir, rows])
        priorities = np.concatenate([priorities, new_keys])
        if len(priorities) > capacity:
            selected = np.argpartition(priorities, capacity - 1)[:capacity]
            reservoir, priorities = reservoir[selected], priorities[selected]
        if stop.requested:
            raise InterruptedError('Scaler sampling interrupted; restart sampling with fixed seed')
    if len(reservoir) < 2:
        raise ValueError('Insufficient finite training rows for scaler')
    scaler = RobustScaler().fit(reservoir)
    state = {'center': scaler.center_.tolist(), 'scale': scaler.scale_.tolist(),
             'features': d['features'], 'sample_rows': len(reservoir), 'eligible_rows': seen,
             'method': 'uniform_priority_reservoir', 'seed': config['seed']}
    save_json(state, cache / 'scaler.json')
    return state


def verify_shards(manifest, cache):
    for shard in manifest['shards']:
        for name, path in shard['files'].items():
            if not (cache / path).exists() or file_sha256(cache / path) != shard['sha256'][name]:
                raise ValueError(f'Corrupt or missing prepared shard: {path}; rebuild into a new cache')


def prepare_data(config, stop):
    """Persist one completed meter at a time; freeze inventory and scaler."""
    freq = validate_config(config)
    d, s = config['data'], config['splits']
    cache = Path(d['cache_directory'])
    groups, inventory = source_inventory(config)
    fingerprint = digest({'format': FORMAT_VERSION, 'data': d, 'splits': s,
                          'seed': config['seed'], 'inventory': inventory})
    with exclusive_lock(cache):
        target = cache / 'manifest.json'
        if target.exists():
            manifest = json.loads(target.read_text())
            if manifest['fingerprint'] != fingerprint:
                raise ValueError('Data/config changed; use a NEW cache directory')
        else:
            manifest = {'format_version': FORMAT_VERSION, 'fingerprint': fingerprint,
                        'inventory': inventory, 'completed_series': [], 'shards': [],
                        'statistics': {}, 'status': 'preparing'}
            save_json(manifest, target)
        if manifest['status'] == 'complete':
            if file_sha256(cache / 'scaler.json') != manifest['scaler_sha256']:
                raise ValueError('Prepared scaler checksum mismatch')
            if d.get('verify_shards', True):
                verify_shards(manifest, cache)
            return manifest, json.loads((cache / 'scaler.json').read_text())
        cut = boundary(s['train_end'], s['timezone'])
        test_start = boundary(s['test_start'], s['timezone'])
        test_end = boundary(s['test_end'], s['timezone'])
        if cut != test_start or test_end <= test_start:
            raise ValueError('Require train_end == test_start < test_end')
        if cut % freq or test_end % freq:
            raise ValueError('Split boundaries must align to the UTC frequency grid')
        for key, paths in sorted(groups.items()):
            if key in manifest['completed_series']:
                continue
            prefix = digest(key)[:24]
            (cache / 'staging').mkdir(parents=True, exist_ok=True)
            db_path = cache / 'staging' / f'{prefix}.sqlite'
            LOG.info('Preparing meter %s', key)
            connection = _stage_series(paths, db_path, config, stop)
            try:
                bounds = connection.execute('SELECT MIN(ts), MAX(ts), COUNT(*) FROM raw').fetchone()
                if bounds[0] is None:
                    segments = [('test', test_start // freq, test_end // freq)]
                    records = _write_segment(connection, key, 'test', test_start // freq,
                                             test_end // freq, config, cache, prefix, stop)
                else:
                    if bounds[0] < 0:
                        raise ValueError('Pre-1970 timestamps are unsupported')
                    first, last = bounds[0] // freq, bounds[1] // freq + 1
                    if last - first > int(d.get('max_series_span_rows', 10_000_000)):
                        raise ValueError(f'{key}: excessive time span; inspect timestamps or increase max_series_span_rows')
                    baseline_end = min(last, cut // freq)
                    n = max(0, baseline_end - first)
                    nv, nc = int(n * s['validation_ratio']), int(n * s['calibration_ratio'])
                    train_end, val_end = baseline_end - nv - nc, baseline_end - nc
                    segments = [('train', first, train_end), ('val', train_end, val_end),
                                ('calibration', val_end, baseline_end),
                                ('test', test_start // freq, test_end // freq)]
                    records = []
                    for kind, a, b in segments:
                        if b > a:
                            records.extend(_write_segment(connection, key, kind, a, b,
                                                          config, cache, prefix, stop))
                manifest['shards'].extend(records)
                manifest['statistics'][key] = {'unique_raw_rows': bounds[2],
                    'invalid_timestamps': connection.execute('SELECT COALESCE(SUM(invalid),0) FROM done').fetchone()[0],
                    'windows': {kind: sum(x['windows'] for x in records if x['split'] == kind)
                                for kind in ('train', 'val', 'calibration', 'test')}}
                manifest['completed_series'].append(key)
                save_json(manifest, target)
            finally:
                connection.close()
            if not d.get('keep_staging', False):
                db_path.unlink(missing_ok=True)
            if stop.requested:
                raise InterruptedError('Preparation stopped at meter boundary')
        for kind in ('train', 'val', 'calibration'):
            if not sum(x['windows'] for x in manifest['shards'] if x['split'] == kind):
                raise ValueError(f'No valid {kind} windows; adjust time splits/window/missing settings')
        if d.get('verify_shards', True):
            verify_shards(manifest, cache)
        scaler = _sample_scaler(manifest, config, cache, stop)
        manifest['scaler_sha256'] = file_sha256(cache / 'scaler.json')
        manifest['status'] = 'complete'
        save_json(manifest, target)
        return manifest, scaler


class MeterWindowDataset(Dataset):
    """One mmap-backed shard, compact indices; no per-window Python objects."""
    def __init__(self, cache, shard, scaler, window_size):
        self.cache, self.shard, self.window = str(cache), shard, int(window_size)
        self.center = np.asarray(scaler['center'], np.float32)
        self.scale = np.asarray(scaler['scale'], np.float32)
        self._open()

    def _open(self):
        root = Path(self.cache)
        self.values = np.load(root / self.shard['files']['values'], mmap_mode='r')
        self.observed = np.load(root / self.shard['files']['observed'], mmap_mode='r')
        self.starts = np.load(root / self.shard['files']['starts'], mmap_mode='r')
        self.valid = np.load(root / self.shard['files']['valid'], mmap_mode='r')
        self.valid_indices = np.flatnonzero(self.valid)

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in ('values', 'observed', 'starts', 'valid', 'valid_indices'):
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start = int(self.starts[self.valid_indices[index]])
        return ((np.asarray(self.values[start:start + self.window]) - self.center) / self.scale).astype(np.float32)


class ResumeBatchSampler:
    """Deterministic shard permutation; cursor counts CONSUMED optimizer batches."""
    def __init__(self, size, batch_size, seed, start_batch=0, shuffle=True):
        self.size, self.batch_size, self.start_batch = size, batch_size, start_batch
        self.indices = np.random.default_rng(seed).permutation(size) if shuffle else np.arange(size)

    def __iter__(self):
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield self.indices[start:start + self.batch_size].tolist()

    def __len__(self):
        return max(0, math.ceil(self.size / self.batch_size) - self.start_batch)

