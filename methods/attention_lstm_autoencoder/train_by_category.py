"""Group channel files by numerically ranked JSON channel IDs and run the existing main.py without modifying it.

Example:
python train_by_category.py --repository meter_repository.json \
    --data-root /content/lstm_large/all_data --config config.yaml \
    --project /content/lstm_large --work-dir /content/category_training \
    --output-root /content/drive/MyDrive/lstm_runs/by_category --prepare-only
Remove --prepare-only to train. Add --resume to resume existing checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import yaml


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--min-channels', type=int, default=20)
    parser.add_argument('--prepare-only', action='store_true', help='Only group files and create configurations; do not run main.py')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.min_channels < 1:
        parser.error('--min-channels must be positive')
    repository_path = Path(args.repository).resolve()
    data_root = Path(args.data_root).resolve()
    config_path = Path(args.config).resolve()
    project = Path(args.project).resolve()
    work = Path(args.work_dir).resolve()
    output = Path(args.output_root).resolve()
    if not data_root.is_dir() or not (project / 'main.py').is_file():
        parser.error('Data directory or project main.py does not exist')
    if work == data_root or data_root in work.parents:
        parser.error('work-dir must be outside data-root to avoid scanning generated links')
    repository = json.loads(repository_path.read_text(encoding='utf-8'))
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    channels = {}
    categories = {}
    for device in repository.values():
        for channel in device.get('channels', []):
            channel_id = str(channel['id'])
            category_id = channel['category_id']
            if isinstance(category_id, bool) or not isinstance(category_id, int):
                raise ValueError('category_id must be an integer')
            name = channel['category_name']
            if category_id in categories and categories[category_id] != name:
                raise ValueError(f'Conflicting names for category {category_id}')
            if channel_id in channels:
                raise ValueError(f'Duplicate channel id: {channel_id}')
            categories[category_id] = name
            channels[channel_id] = category_id
    # Rank existing channels numerically: C1,C2,C6 -> file suffixes 0,1,2.
    file_id_to_channel = {}
    device_mapping = []
    for device_key, device in repository.items():
        device_id = str(device.get('device_id', device_key))
        ranked = []
        for channel in device.get('channels', []):
            channel_id = str(channel['id'])
            match = re.fullmatch(re.escape(device_id) + r'_C(\d+)', channel_id)
            if not match:
                raise ValueError(f'Invalid channel ID for device {device_id}: {channel_id}')
            ranked.append((int(match.group(1)), channel))
        ranked.sort(key=lambda item: item[0])
        if len({number for number, _ in ranked}) != len(ranked):
            raise ValueError(f'Duplicate numeric channel numbers: {device_id}')
        for index, (_, channel) in enumerate(ranked):
            file_id = f'{device_id}_{index}'
            if file_id in file_id_to_channel:
                raise ValueError(f'Duplicate file mapping: {file_id}')
            file_id_to_channel[file_id] = str(channel['id'])
            device_mapping.append({'device_id': device_id, 'csv_stem': file_id,
                                   'channel_id': str(channel['id']),
                                   'category_id': channel['category_id'],
                                   'category_name': channel['category_name'],
                                   'device_channel_count': len(ranked),
                                   'device_category_count': len({c['category_id'] for _, c in ranked})})
    extensions = {e.lower() for e in config['data'].get('extensions', ['.csv', '.parquet'])}
    matched = {}
    unmatched = []
    for path in sorted(data_root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.name.startswith('_'):
            continue
        channel_id = file_id_to_channel.get(path.stem)
        # Exact original JSON channel filenames are also supported.
        if channel_id is None and path.stem in channels:
            channel_id = path.stem
        if channel_id is None:
            unmatched.append(str(path))
            continue
        if channel_id in matched:
            raise ValueError(f'Multiple data files map to channel {channel_id}; resolve duplicates first')
        matched[channel_id] = path.resolve()
    write_json(output / 'device_channel_mapping.json', device_mapping)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'matching_report.json', {
        'unmatched_files': unmatched,
        'channels_without_files': sorted(set(channels) - set(matched)),
        'matched_channels': len(matched),
        'minimum_channels': args.min_channels,
        'count_rule': 'Unique JSON channels with matching input files using numeric ordering of existing JSON channels',
    })
    plans = []
    for category_id, name in sorted(categories.items()):
        registered = [cid for cid, cat in channels.items() if cat == category_id]
        selected = sorted(cid for cid in registered if cid in matched)
        plan = {'category_id': category_id, 'category_name': name,
                'registered_channels': len(registered), 'matched_channels': len(selected),
                'channel_ids': selected, 'matching_mode': 'rank_existing_channels_numeric_C_order',
                'count_unit': 'matched_json_channels',
                'file_to_channel': {matched[cid].name: cid for cid in selected}}
        if len(selected) < args.min_channels:
            plan['status'] = 'skipped_below_minimum'
            plans.append(plan)
            print(f'SKIP {category_id} {name}: {len(selected)} matched channels', flush=True)
            continue
        category_work = work / f'category_{category_id}'
        category_output = output / f'category_{category_id}'
        inputs = category_work / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        inventory = [{'id': cid, 'path': str(matched[cid]),
                      'size': matched[cid].stat().st_size,
                      'mtime_ns': matched[cid].stat().st_mtime_ns} for cid in selected]
        signature = hashlib.sha256(json.dumps({'category': plan, 'files': inventory, 'config': config}, sort_keys=True).encode()).hexdigest()
        marker = category_work / 'group_manifest.json'
        if marker.exists():
            old = json.loads(marker.read_text(encoding='utf-8'))
            if old['signature'] != signature:
                raise ValueError(f'Category {category_id} inputs/config changed; use new work/output directories')
        elif category_output.exists() and any(category_output.iterdir()):
            raise ValueError(f'Existing output without group manifest: {category_output}; use a new output directory')
        expected = {matched[cid].name for cid in selected}
        if any(p.name not in expected for p in inputs.iterdir()):
            raise ValueError(f'Unexpected files in {inputs}; use a new work directory')
        for cid in selected:
            source = matched[cid]
            link = inputs / source.name
            if link.is_symlink():
                if link.resolve() != source:
                    raise ValueError(f'Incorrect link: {link}')
            elif link.exists():
                raise ValueError(f'Refusing to overwrite {link}')
            else:
                link.symlink_to(source)
        category_config = json.loads(json.dumps(config))
        category_config['data'].update(root=str(inputs), cache_directory=str(category_work / 'cache'),
                                       input_view_directory=str(category_work / 'input_view'),
                                       meter_limit=None, series_id_regex=None)
        category_config['output']['directory'] = str(category_output)
        category_work.mkdir(parents=True, exist_ok=True)
        category_config_path = category_work / 'config.yaml'
        category_config_path.write_text(yaml.safe_dump(category_config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        write_json(marker, {'signature': signature, 'files': inventory, **plan})
        write_json(category_output / 'category_info.json', plan)
        plan.update(config_path=str(category_config_path), output_directory=str(category_output), status='ready')
        plans.append(plan)
    write_json(output / 'category_training_summary.json', plans)
    if args.prepare_only:
        print('Grouping complete. Remove --prepare-only to train.', flush=True)
        return
    for plan in plans:
        if plan['status'] != 'ready':
            continue
        command = [sys.executable, '-u', str(project / 'main.py'), 'train', '--config', plan['config_path']]
        checkpoint = Path(plan['output_directory']) / 'last_checkpoint.json'
        if checkpoint.exists():
            if not args.resume:
                raise ValueError('Existing checkpoint: rerun with --resume or use new directories')
            command.append('--resume')
        plan['status'] = 'running'
        write_json(output / 'category_training_summary.json', plans)
        print(f"TRAIN {plan['category_id']} {plan['category_name']}", flush=True)
        result = subprocess.run(command, cwd=project)
        plan['status'] = 'completed' if result.returncode == 0 else ('interrupted' if result.returncode == 75 else 'failed')
        plan['returncode'] = result.returncode
        write_json(output / 'category_training_summary.json', plans)
        if result.returncode:
            sys.exit(result.returncode)
        # Existing export command preserves the pipeline's output contract.
        result = subprocess.run([sys.executable, '-u', str(project / 'main.py'), 'export', '--config', plan['config_path']], cwd=project) if config['training'].get('predict_after_training', False) else None
        if result is not None and result.returncode:
            plan['status'] = 'export_failed'
            write_json(output / 'category_training_summary.json', plans)
            sys.exit(result.returncode)


if __name__ == '__main__':
    main()

