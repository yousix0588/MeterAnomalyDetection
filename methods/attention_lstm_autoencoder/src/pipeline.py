"""Shard-wise training with transactionally committed resume checkpoints."""
from __future__ import annotations
import json
import logging
import math
import os
import platform
import shutil
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import MeterWindowDataset, ResumeBatchSampler, prepare_data, validate_config
from .model import LSTMAutoencoder, reconstruction_errors, reconstruction_point_errors
from .utils import (StopRequest, atomic_target, digest, exclusive_lock, file_sha256,
                    random_state, restore_random_state, save_json, save_numpy,
                    save_torch, select_device, set_seed)

LOG = logging.getLogger(__name__)
CODE_VERSION = 'large-shards-v1'


def _loader(dataset, config, seed=0, start_batch=0, shuffle=False):
    t = config['training']
    workers = int(t.get('num_workers', 0))
    kwargs = {'num_workers': workers, 'pin_memory': t.get('pin_memory', True),
              'generator': torch.Generator().manual_seed(seed)}
    if workers:
        kwargs['prefetch_factor'] = int(t.get('prefetch_factor', 2))
        kwargs['persistent_workers'] = False
    return DataLoader(dataset, batch_sampler=ResumeBatchSampler(len(dataset), int(t['batch_size']),
                      seed, start_batch, shuffle), **kwargs)


def _dataset(shard, manifest, scaler, config):
    return MeterWindowDataset(config['data']['cache_directory'], shard, scaler,
                              config['data']['window_size'])


def _model(config, device):
    return LSTMAutoencoder(input_size=len(config['data']['features']), **config['model']).to(device)


def _seed(config, epoch, shard_index):
    return int(digest([config['seed'], epoch, shard_index])[:15], 16)


def _identity(config, manifest):
    # Output/device/worker settings may change; scientific/training settings may not.
    training = {k: v for k, v in config['training'].items() if k not in
                ('device', 'num_workers', 'prefetch_factor', 'pin_memory',
                 'checkpoint_seconds', 'checkpoint_steps', 'keep_epoch_checkpoints', 'keep_shard_checkpoints')}
    return digest({'code': CODE_VERSION, 'manifest': manifest['fingerprint'],
                   'scaler': manifest['scaler_sha256'], 'model': config['model'],
                   'training': training, 'seed': config['seed'],
                   'source_sha256': {p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')}})


def _commit(output, state, model, optimizer, amp_scaler, identity, config, archive=False):
    """Immutable payload + atomic pointer, retaining the previous valid commit."""
    folder = output / 'checkpoints'
    folder.mkdir(exist_ok=True)
    name = f'step_{state["global_step"]:012d}_{uuid.uuid4().hex}.pt'
    value = {'version': CODE_VERSION, 'identity': identity, 'state': state,
             'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
             'amp_scaler_state': amp_scaler.state_dict(), 'amp_enabled': amp_scaler.is_enabled(),
             'random_state': random_state(),
             'config': config}
    save_torch(value, folder / name)
    pointer_file = output / 'last_checkpoint.json'
    previous = json.loads(pointer_file.read_text()).get('current') if pointer_file.exists() else None
    current = {'file': f'checkpoints/{name}', 'sha256': file_sha256(folder / name)}
    save_json({'current': current, 'previous': previous}, pointer_file)
    if archive:
        archive_path = (output / archive) if isinstance(archive, str) else (output / 'epochs' / f'epoch_{state["epoch"]:04d}.json')
        save_json(current, archive_path)
    retained = {current['file']}
    if previous:
        retained.add(previous['file'])
    for directory in (output / 'epochs', output / 'stage_checkpoints'):
        for path in directory.glob('*.json') if directory.exists() else []:
            retained.add(json.loads(path.read_text())['file'])
    for path in folder.glob('step_*.pt'):
        if str(path.relative_to(output)) not in retained:
            path.unlink()
    save_json(state, output / 'training_state.json')
    save_json(state['history'], output / 'training_history.json')
    with atomic_target(output / 'training_history.csv') as temporary:
        pd.DataFrame(state['history']).to_csv(temporary, index=False)


def _restore(output, identity):
    pointer = json.loads((output / 'last_checkpoint.json').read_text())
    failures = []
    for item in (pointer.get('current'), pointer.get('previous')):
        if item is None:
            continue
        path = output / item['file']
        try:
            if file_sha256(path) != item['sha256']:
                raise ValueError('checksum mismatch')
            # Only load checkpoints created by this project from a trusted directory.
            value = torch.load(path, map_location='cpu', weights_only=False)
        except (OSError, ValueError, RuntimeError, EOFError) as exc:
            failures.append(str(exc))
            continue
        if value['identity'] != identity:
            raise ValueError('Resume configuration/data/code differs from checkpoint')
        if failures:
            LOG.warning('Latest checkpoint unusable; restored previous commit: %s', path)
        return value
    raise ValueError(f'No usable checkpoint: {failures}')


@torch.no_grad()
def score_dataset(model, loader, device):
    """Stream validation statistics instead of concatenating all scores."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        values = reconstruction_errors(batch, model(batch)).float()
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite validation errors')
        total += values.double().sum().item()
        count += len(values)
    return total, count


def _validate(model, manifest, scaler, config, device, stop):
    total, count = 0.0, 0
    for shard in manifest['shards']:
        if shard['split'] != 'val' or not shard['windows']:
            continue
        dataset = _dataset(shard, manifest, scaler, config)
        # Check stop requests at every validation batch.
        for batch in _loader(dataset, config):
            with torch.no_grad():
                model.eval()
                x = batch.to(device, non_blocking=True)
                scores = reconstruction_errors(x, model(x)).float()
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite validation errors')
                total += scores.double().sum().item()
                count += len(scores)
            if stop.requested:
                raise InterruptedError('Validation interrupted; repeat validation on resume')
    if not count:
        raise ValueError('Empty validation set')
    return total / count


@torch.no_grad()
def score_interval_blocks(model, loader, device, block_size, stop=None):
    """One bounded shard of scores. Inference uses float32 consistently."""
    model.eval()
    maxima, means, deviations = [], [], []
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        error = reconstruction_point_errors(x, model(x))[:, -block_size:].float()
        if not torch.isfinite(error).all():
            raise FloatingPointError('Nonfinite scoring errors')
        maxima.append(error.max(dim=1).values.cpu().numpy())
        means.append(error.mean(dim=1).cpu().numpy())
        deviations.append(error.std(dim=1, unbiased=False).cpu().numpy())
        if stop is not None and stop.requested:
            raise InterruptedError('Scoring interrupted; repeat unfinished shard on resume')
    return {name: np.concatenate(values) if values else np.empty(0, np.float32)
            for name, values in [('max', maxima), ('mean', means), ('std', deviations)]}


def frozen_ecdf(values, baseline):
    """baseline must be sorted; sample ECDF is explicitly approximate."""
    values = np.asarray(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    result[finite] = np.searchsorted(baseline, values[finite], side='right') / len(baseline)
    return result


def _score_binding(identity, best_sha, manifest, config):
    return digest([identity, best_sha, manifest['fingerprint'], manifest['scaler_sha256'],
                   config['data']['score_block_size'], 'last_block_point_mse_max'])


def calibrate(config, manifest, scaler, model, device, output, binding, stop):
    """Resumable uniform score reservoir; no complete score distribution in RAM."""
    directory = output / 'calibration'
    directory.mkdir(exist_ok=True)
    state_file = directory / 'progress.pt'
    k = int(config['training'].get('calibration_max_scores', 1_000_000))
    if k < 2:
        raise ValueError('calibration_max_scores must be >= 2')
    if state_file.exists():
        progress = torch.load(state_file, map_location='cpu', weights_only=False)
        if progress['binding'] != binding:
            raise ValueError('Calibration belongs to another model/data version')
    else:
        rng = np.random.default_rng(config['seed'] + 17)
        progress = {'binding': binding, 'cursor': 0, 'seen': 0,
                    'keys': np.empty(0), 'max': np.empty(0, np.float32),
                    'mean': np.empty(0, np.float32), 'rng': rng.bit_generator.state}
    shards = [s for s in manifest['shards'] if s['split'] == 'calibration']
    rng = np.random.default_rng()
    rng.bit_generator.state = progress['rng']
    for index in range(progress['cursor'], len(shards)):
        shard = shards[index]
        dataset = _dataset(shard, manifest, scaler, config)
        scores = score_interval_blocks(model, _loader(dataset, config), device,
                                       config['data']['score_block_size'], stop)
        keys = np.concatenate([progress['keys'], rng.random(len(scores['max']))])
        maximum = np.concatenate([progress['max'], scores['max']])
        mean = np.concatenate([progress['mean'], scores['mean']])
        if len(keys) > k:
            selected = np.argpartition(keys, k - 1)[:k]
            keys, maximum, mean = keys[selected], maximum[selected], mean[selected]
        progress.update(cursor=index + 1, seen=progress['seen'] + len(scores['max']),
                        keys=keys, max=maximum, mean=mean, rng=rng.bit_generator.state)
        save_torch(progress, state_file)
        if stop.requested:
            raise InterruptedError('Calibration stopped at committed shard')
    if not len(progress['max']):
        raise ValueError('No calibration scores')
    maximum, mean = np.sort(progress['max']), np.sort(progress['mean'])
    save_numpy(maximum, directory / 'max_sorted.npy')
    save_numpy(mean, directory / 'mean_sorted.npy')
    metadata = {'binding': binding, 'threshold': float(np.quantile(maximum, config['training']['threshold_quantile'])),
                'threshold_quantile': config['training']['threshold_quantile'],
                'score_definition': 'last_block_point_reconstruction_mse_max',
                'score_block_size': config['data']['score_block_size'],
                'frequency': config['data']['frequency'], 'source': 'held_out_calibration',
                'method': 'uniform_priority_reservoir_quantile_and_ecdf',
                'total_scores': progress['seen'], 'sample_scores': len(maximum),
                'approximate': progress['seen'] > len(maximum),
                'ecdf_dkw_epsilon_95': min(1.0, math.sqrt(math.log(40) / (2 * len(maximum)))),
                'ecdf_bound_note': 'Sampling error only, conditional on the fixed calibration scores; not a population false-alarm guarantee',
                'max_sample_sha256': file_sha256(directory / 'max_sorted.npy'),
                'mean_sample_sha256': file_sha256(directory / 'mean_sorted.npy')}
    save_json(metadata, output / 'metadata.json')
    return metadata, maximum, mean


def _prediction_frame(dataset, scores, metadata, maximum, mean, config):
    """Emit candidates even when unavailable; original observation coverage retained."""
    shard = dataset.shard
    d = config['data']
    candidates = len(dataset.starts)
    fields = {name: np.full(candidates, np.nan) for name in ('max', 'mean', 'std')}
    for name in fields:
        fields[name][dataset.valid_indices] = scores[name]
    starts = np.asarray(dataset.starts)
    freq = pd.to_timedelta(d['frequency']).value
    ends = (shard['start_bin'] + starts + dataset.window) * freq
    block = d['score_block_size']
    cumulative = np.r_[0, np.cumsum(dataset.observed, dtype=np.int64)]
    counts = cumulative[starts + dataset.window] - cumulative[np.maximum(0, starts + dataset.window - block)]
    available = np.asarray(dataset.valid)
    result = pd.DataFrame({
        'model': 'attention_lstm_autoencoder', 'series_id': shard['series_id'],
        'interval_start': pd.to_datetime(ends - block * freq, utc=True),
        'interval_end': pd.to_datetime(ends, utc=True),
        'window_start': pd.to_datetime((shard['start_bin'] + starts) * freq, utc=True),
        'window_end': pd.to_datetime(ends, utc=True),
        'raw_score': fields['max'], 'anomaly_score': fields['max'],
        'max_score': fields['max'], 'mean_score': fields['mean'], 'score_std': fields['std'],
        'max_percentile': frozen_ecdf(fields['max'], maximum),
        'mean_percentile': frozen_ecdf(fields['mean'], mean),
        'threshold': metadata['threshold'], 'available': available,
        'valid_point_count': counts, 'expected_point_count': block,
        'coverage_ratio': counts / block,
        'source_resolution_minutes': freq / 60_000_000_000,
        'aggregation_method': f'last_{block}_points_of_{dataset.window}_point_window',
        'data_status': np.where(available, np.where(counts == block, 'observed', 'interpolated'),
                                np.where(starts < 0, 'insufficient_history', 'missing_data')),
        'calibration_version': metadata['binding'], 'model_version': metadata['binding']})
    anomalies = pd.array(fields['max'] > metadata['threshold'], dtype='boolean')
    anomalies[~available] = pd.NA
    result['is_anomaly'] = anomalies
    return result


def predict(config, manifest, scaler, model, device, output, metadata, maximum, mean, stop):
    """Commit independent CSV shards plus completion ledger; resume without duplicates."""
    directory = output / 'predictions'
    directory.mkdir(exist_ok=True)
    ledger_file = directory / 'manifest.json'
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {
        'binding': metadata['binding'], 'completed': {}, 'status': 'running'}
    if ledger['binding'] != metadata['binding']:
        raise ValueError('Prediction output belongs to another model')
    for index, shard in enumerate(manifest['shards']):
        if shard['split'] != 'test':
            continue
        key = str(index)
        name = f'scores_{index:08d}.csv'
        entry = ledger['completed'].get(key)
        if entry and (directory / name).exists() and file_sha256(directory / name) == entry['sha256']:
            continue
        dataset = _dataset(shard, manifest, scaler, config)
        scores = score_interval_blocks(model, _loader(dataset, config), device,
                                       config['data']['score_block_size'], stop)
        frame = _prediction_frame(dataset, scores, metadata, maximum, mean, config)
        with atomic_target(directory / name) as temporary:
            frame.to_csv(temporary, index=False)
        valid = frame[frame['available']]
        ledger['completed'][key] = {'file': name, 'sha256': file_sha256(directory / name),
            'series_id': shard['series_id'], 'candidates': len(frame), 'available': len(valid),
            'anomalies': int(valid['is_anomaly'].sum()),
            'score_sum': float(valid['raw_score'].sum()),
            'max_score': float(valid['raw_score'].max()) if len(valid) else None}
        save_json(ledger, ledger_file)
        if stop.requested:
            raise InterruptedError('Prediction stopped after committed shard')
    summary = {}
    for entry in ledger['completed'].values():
        row = summary.setdefault(entry['series_id'], {'series_id': entry['series_id'],
                    'windows': 0, 'available_windows': 0, 'anomalies': 0, 'score_sum': 0.0, 'max_score': None})
        row['windows'] += entry['candidates']
        row['available_windows'] += entry['available']
        row['anomalies'] += entry['anomalies']
        row['score_sum'] += entry['score_sum']
        if entry['max_score'] is not None:
            row['max_score'] = max(row['max_score'] if row['max_score'] is not None else -np.inf, entry['max_score'])
    for row in summary.values():
        row['mean_score'] = row.pop('score_sum') / row['available_windows'] if row['available_windows'] else None
    with atomic_target(output / 'meter_summary.csv') as temporary:
        pd.DataFrame(summary.values()).to_csv(temporary, index=False)
    ledger['status'] = 'complete'
    save_json(ledger, ledger_file)
    return directory


def _finish(config, manifest, scaler, state, device, output, identity, stop):
    best = output / state['best_file']
    if file_sha256(best) != state['best_sha256']:
        raise ValueError('Best model checksum mismatch')
    saved = torch.load(best, map_location='cpu', weights_only=False)
    model = _model(config, device)
    model.load_state_dict(saved['model_state'])
    # Convenient stable inference file; reference to immutable best remains in state.
    with atomic_target(output / 'best_model.pt') as temporary:
        shutil.copyfile(best, temporary)
    binding = _score_binding(identity, state['best_sha256'], manifest, config)
    metadata, maximum, mean = calibrate(config, manifest, scaler, model, device, output, binding, stop)
    metadata.update(model_file=state['best_file'], model_sha256=state['best_sha256'],
                    manifest_fingerprint=manifest['fingerprint'], scaler_sha256=manifest['scaler_sha256'],
                    best_epoch=state['best_epoch'], best_val_loss=state['best_loss'],
                    split_config=config['splits'], data_statistics=manifest['statistics'])
    save_json(metadata, output / 'metadata.json')
    if config['training'].get('predict_after_training', True):
        predict(config, manifest, scaler, model, device, output, metadata, maximum, mean, stop)


def _complete_stage(config, manifest, scaler, state, device, output, identity, stop):
    """Optional stage model and its independently bound calibration artifacts."""
    directory = output / 'stages' / state['pending_stage']
    stage_path = directory / 'model.pt'
    saved = torch.load(stage_path, map_location='cpu', weights_only=False)
    stage_model = _model(config, device)
    stage_model.load_state_dict(saved['model_state'])
    binding = _score_binding(identity, file_sha256(stage_path), manifest, config)
    metadata, _, _ = calibrate(config, manifest, scaler, stage_model, device, directory, binding, stop)
    metadata.update(model_file='model.pt', model_sha256=file_sha256(stage_path),
                    global_step=state['global_step'], epoch=state['epoch'] + 1,
                    shard_position=state['shard_position'])
    save_json(metadata, directory / 'metadata.json')
    state['pending_stage'] = None


def train(config, resume=False):
    """Continue one model over all shards; each epoch visits the complete training set."""
    validate_config(config)
    output = Path(config['output']['directory'])
    with exclusive_lock(output), StopRequest() as stop:
        logger = logging.FileHandler(output / 'run.log', encoding='utf-8')
        logging.getLogger().addHandler(logger)
        try:
            return _train_locked(config, output, resume, stop)
        except InterruptedError as exc:
            save_json({'status': 'interrupted', 'reason': stop.reason or str(exc)}, output / 'run_status.json')
            LOG.warning('%s', exc)
            raise
        except BaseException as exc:
            save_json({'status': 'failed', 'reason': repr(exc),
                       'resume': 'Restore the latest valid committed checkpoint; do not load a partially updated model'},
                      output / 'run_status.json')
            LOG.exception('Run failed')
            raise
        finally:
            logging.getLogger().removeHandler(logger)
            logger.close()


def _train_locked(config, output, resume, stop):
    existing = (output / 'last_checkpoint.json').exists()
    if existing and not resume:
        raise ValueError('Existing run; use --resume or choose a new output directory')
    if resume and not existing:
        raise ValueError('No committed training checkpoint; run without --resume to continue preparation')
    set_seed(config['seed'])
    manifest, scaler = prepare_data(config, stop)
    identity = _identity(config, manifest)
    device = select_device(config['training']['device'])
    t = config['training']
    amp = bool(t.get('mixed_precision', True)) and device.type == 'cuda'
    model = _model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    state = {'phase': 'train', 'epoch': 0, 'shard_position': 0, 'batch_position': 0,
             'global_step': 0, 'loss_sum': 0.0, 'samples': 0, 'history': [],
             'best_loss': float('inf'), 'best_file': None, 'best_sha256': None,
             'best_epoch': None, 'bad_epochs': 0, 'resume_count': 0, 'pending_stage': None}
    if resume:
        value = _restore(output, identity)
        state = value['state']
        if value['amp_enabled'] != amp:
            raise ValueError('Effective mixed precision changed; resume on a compatible device')
        model.load_state_dict(value['model_state'])
        optimizer.load_state_dict(value['optimizer_state'])
        grad_scaler.load_state_dict(value['amp_scaler_state'])
        restore_random_state(value['random_state'])
        state['resume_count'] += 1
    else:
        save_json(config, output / 'config.json')
        save_json(scaler, output / 'scaler.json')
        save_json({'code_version': CODE_VERSION, 'python': platform.python_version(),
                   'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                   'cuda_runtime': torch.version.cuda, 'device': str(device),
                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                   'identity': identity, 'manifest_fingerprint': manifest['fingerprint']}, output / 'environment.json')
        _commit(output, state, model, optimizer, grad_scaler, identity, config)
    save_json({'status': 'running', 'identity': identity}, output / 'run_status.json')
    shards = [s for s in manifest['shards'] if s['split'] == 'train' and s['windows']]
    last_save = time.monotonic()
    if state.get('pending_stage'):
        _complete_stage(config, manifest, scaler, state, device, output, identity, stop)
        _commit(output, state, model, optimizer, grad_scaler, identity, config)
    while state['phase'] in ('train', 'validate'):
        epoch = state['epoch']
        order = np.random.default_rng(_seed(config, epoch, -1)).permutation(len(shards))
        if state['phase'] == 'train':
            model.train()
            for position in range(state['shard_position'], len(order)):
                shard_index = int(order[position])
                shard = shards[shard_index]
                dataset = _dataset(shard, manifest, scaler, config)
                start_batch = state['batch_position'] if position == state['shard_position'] else 0
                loader = _loader(dataset, config, _seed(config, epoch, shard_index), start_batch, True)
                LOG.info('epoch=%d shard=%d/%d batch_start=%d windows=%d', epoch + 1, position + 1, len(order), start_batch, len(dataset))
                for batch_index, batch in enumerate(loader, start=start_batch):
                    if stop.requested:
                        _commit(output, state, model, optimizer, grad_scaler, identity, config)
                        raise InterruptedError('Stopped before next optimizer update')
                    x = batch.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                        reconstructed = model(x)
                        loss = ((reconstructed.float() - x.float()) ** 2).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss; restore last committed state')
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    norm = nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'])
                    if not torch.isfinite(norm):
                        if not amp:
                            raise FloatingPointError('Nonfinite gradients')
                        # GradScaler skips the update and adjusts scale; cursor still consumes this batch.
                        LOG.warning('AMP gradient overflow: skipped optimizer update')
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    state['global_step'] += 1
                    state['batch_position'] = batch_index + 1
                    state['loss_sum'] += float(loss.detach()) * len(x)
                    state['samples'] += len(x)
                    steps = int(t.get('checkpoint_steps', 1000))
                    seconds = float(t.get('checkpoint_seconds', 600))
                    due = ((steps > 0 and state['global_step'] % steps == 0) or
                           (seconds > 0 and time.monotonic() - last_save >= seconds))
                    if due or stop.requested:
                        _commit(output, state, model, optimizer, grad_scaler, identity, config)
                        last_save = time.monotonic()
                    if stop.requested:
                        raise InterruptedError('Stopped after completed optimizer batch')
                state['shard_position'], state['batch_position'] = position + 1, 0
                stage_name = f'epoch_{epoch + 1:04d}_shard_{position + 1:06d}'
                if t.get('calibrate_shard_checkpoints', False):
                    stage_dir = output / 'stages' / stage_name
                    save_torch({'model_state': model.state_dict(), 'config': config,
                                'identity': identity}, stage_dir / 'model.pt')
                    save_json(scaler, stage_dir / 'scaler.json')
                    save_json(config, stage_dir / 'config.json')
                    state['pending_stage'] = stage_name
                archive = (f'stage_checkpoints/{stage_name}.json'
                           if t.get('keep_shard_checkpoints', True) else False)
                _commit(output, state, model, optimizer, grad_scaler, identity, config, archive=archive)
                if state.get('pending_stage'):
                    _complete_stage(config, manifest, scaler, state, device, output, identity, stop)
                    _commit(output, state, model, optimizer, grad_scaler, identity, config)
                last_save = time.monotonic()
            state['phase'] = 'validate'
            _commit(output, state, model, optimizer, grad_scaler, identity, config)
        val_loss = _validate(model, manifest, scaler, config, device, stop)
        if val_loss < state['best_loss']:
            name = f'models/best_epoch_{epoch + 1:04d}_{uuid.uuid4().hex}.pt'
            save_torch({'model_state': model.state_dict(), 'config': config, 'identity': identity,
                        'epoch': epoch + 1, 'val_loss': val_loss}, output / name)
            state.update(best_loss=val_loss, best_file=name, best_sha256=file_sha256(output / name),
                         best_epoch=epoch + 1, bad_epochs=0)
        else:
            state['bad_epochs'] += 1
        row = {'epoch': epoch + 1, 'train_loss': state['loss_sum'] / state['samples'],
               'val_loss': val_loss, 'global_step': state['global_step'], 'train_windows': state['samples']}
        state['history'].append(row)
        LOG.info('epoch=%d train=%.8f validation=%.8f', epoch + 1, row['train_loss'], val_loss)
        state.update(epoch=epoch + 1, shard_position=0, batch_position=0, loss_sum=0.0, samples=0)
        state['phase'] = 'finalize' if state['epoch'] >= t['epochs'] or state['bad_epochs'] >= t['patience'] else 'train'
        _commit(output, state, model, optimizer, grad_scaler, identity, config,
                archive=t.get('keep_epoch_checkpoints', False))
    if state['phase'] == 'finalize':
        _finish(config, manifest, scaler, state, device, output, identity, stop)
        state['phase'] = 'complete'
        _commit(output, state, model, optimizer, grad_scaler, identity, config)
    save_json({'status': 'complete', 'identity': identity, 'best_epoch': state['best_epoch']}, output / 'run_status.json')
    return output


def predict_saved(config):
    """Rerun/resume prediction for the same frozen prepared test dataset."""
    output = Path(config['output']['directory'])
    with exclusive_lock(output), StopRequest() as stop:
        manifest, scaler = prepare_data(config, stop)
        identity = _identity(config, manifest)
        value = _restore(output, identity)
        if value['state']['phase'] not in ('finalize', 'complete'):
            raise ValueError('Training is not finished; use train --resume first')
        _finish(config, manifest, scaler, value['state'], select_device(config['training']['device']),
                output, identity, stop)
    return output / 'predictions'


def export_scores(config):
    """Optional streaming single CSV for downstream ensemble evaluation scripts."""
    output = Path(config['output']['directory'])
    with exclusive_lock(output):
        directory = output / 'predictions'
        ledger = json.loads((directory / 'manifest.json').read_text())
        if ledger['status'] != 'complete':
            raise ValueError('Finish prediction before exporting')
        target = output / 'scores_15min.csv'
        with atomic_target(target) as temporary:
            with temporary.open('wb') as destination:
                wrote_header = False
                for key in sorted(ledger['completed'], key=int):
                    entry = ledger['completed'][key]
                    path = directory / entry['file']
                    if file_sha256(path) != entry['sha256']:
                        raise ValueError(f'Prediction shard checksum mismatch: {path}')
                    with path.open('rb') as source:
                        header = source.readline()
                        if not wrote_header:
                            destination.write(header)
                            wrote_header = True
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return target

