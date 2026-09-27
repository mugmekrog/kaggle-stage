"""Recover the supplied RF notebook split, accepting only its exact cached validation IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random


def verified_split(all_ids, cached_ids):
    ids = sorted(all_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate source image IDs')
    # Exact recipe in the user-supplied astra_rfdetr_4h (2).ipynb.
    blocks = [ids[i:i + 25] for i in range(0, len(ids), 25)]
    random.Random(42).shuffle(blocks)
    n_val_blocks = max(1, round(len(blocks) * 0.05))
    validation = {key for block in blocks[:n_val_blocks] for key in block}
    cached_ids = set(cached_ids)
    if cached_ids != validation:
        raise ValueError('RF cache IDs do not exactly match the supplied notebook split. '
                         f'Expected={len(validation)}, cached={len(cached_ids)}, '
                         f'missing={len(validation - cached_ids)}, extra={len(cached_ids - validation)}. '
                         'No split exported; do not use a partial cache as a training exclusion list.')
    return set(ids) - validation, validation


def recover(prepared, dataset, cache, output):
    manifest_path = Path(prepared) / 'manifest.json'
    info = json.loads(manifest_path.read_text(encoding='utf-8'))
    all_ids = info['train_ids'] + info['valid_ids']
    image_paths = [p for p in (Path(dataset) / 'train/images').iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if len(image_paths) != len(all_ids) or {p.stem for p in image_paths} != set(all_ids):
        raise ValueError('Prepared IDs and source train image IDs differ')
    print('Reading your RF validation cache...', flush=True)
    # This is the user's own trusted prediction cache, not a third-party pickle.
    with Path(cache).open('rb') as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise ValueError('Expected RF cache dictionary keyed by original image ID')
    training, validation = verified_split(all_ids, raw.keys())
    records = {r['id']: r for r in info['records']}
    for key, value in raw.items():
        if tuple(value['wh']) != (records[key]['width'], records[key]['height']):
            raise ValueError(f'RF cache dimensions differ from source data: {key}')
        if not {'tile', 'tile_flip', 'full', 'full_flip'} <= set(value['views']):
            raise ValueError(f'Missing RF prediction views: {key}')
    del raw
    with Path(cache).open('rb') as handle:
        cache_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    common = set(info['valid_ids']) & validation
    counts = {name: sum(b[0] == c for key in common for b in records[key]['boxes'])
              for c, name in enumerate(info['classes'])}
    if not common or not all(counts.values()):
        raise ValueError('Common validation needs examples of all four classes')
    result = dict(images=[dict(id=i, file_name=key + '.jpg')
                          for i, key in enumerate(sorted(training))],
                  original_train_images=len(training), validation_ids=sorted(validation),
                  source='Supplied RF notebook recipe, exactly matched to validation cache IDs',
                  recipe=dict(seed=42, block_size=25, validation_fraction=0.05),
                  cache=str(cache), cache_sha256=cache_hash,
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  common_validation_images=len(common), common_class_counts=counts,
                  note='Reconstructed split, not an archived training manifest. Assumes the supplied '
                       'notebook produced this cache and checkpoint without changing training membership.')
    output = Path(output)
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if {Path(r['file_name']).stem for r in previous['images']} != training:
            raise ValueError('Existing saved RF split differs; refusing to overwrite')
        print('Existing split has identical training IDs.', flush=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'RF train={len(training)} / validation={len(validation)}', flush=True)
    print(f'Common validation={len(common)}; class counts={counts}', flush=True)
    print('SPLIT READY:', output, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'dataset', 'cache', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    recover(**vars(parser.parse_args()))
