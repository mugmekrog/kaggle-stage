"""Recover evaluation metadata only, requiring the exact fingerprint saved during training."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from common.predictions import image_files
from common.runtime import CLASSES
from prepare_data import split_images
from yolo26l_custom.config import split_fingerprint


def recover(dataset, run, output, candidates=()):
    dataset, run, output = Path(dataset), Path(run), Path(output)
    meta = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    expected = meta.get('split_fingerprint')
    if not expected:
        raise ValueError('Training run.json has no split fingerprint; cannot verify recovered metadata')
    for folder in candidates:
        path = Path(folder) / 'manifest.json'
        if path.is_file():
            info = json.loads(path.read_text(encoding='utf-8'))
            if split_fingerprint(info) == expected:
                print('Matching existing manifest:', path, flush=True)
                return path.parent
    if output.exists():
        raise FileExistsError('Use a new metadata output directory')
    files = image_files(dataset / 'train/images')
    annotations = dataset / 'train/annotations.csv'
    if not annotations.is_file() or not files:
        raise FileNotFoundError(f'Original training images/annotations missing under {dataset}')
    boxes, seen = defaultdict(list), set()
    with annotations.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['image_id']
            if key not in files or row['label'] not in CLASSES:
                raise ValueError('Unknown source image or class')
            x, y, w, h = (float(row[k]) for k in ('x', 'y', 'w', 'h'))
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or w*h < 200:
                raise ValueError('Invalid source box')
            entry = (key, CLASSES.index(row['label']), x, y, w, h)
            if entry not in seen:
                seen.add(entry)
                boxes[key].append(entry[1:])
    lists = [run / (split + '_images.txt') for split in ('train', 'valid')]
    if all(p.is_file() for p in lists):
        train, valid = [[Path(line.strip().replace('\\', '/')).stem
                         for line in p.read_text(encoding='utf-8').splitlines() if line.strip()] for p in lists]
        recipe = 'Archived training image lists'
    else:
        print('Archived image lists absent; checking original preparation recipe against training fingerprint.', flush=True)
        train, valid = split_images(files, boxes, 0.2, 42)
        recipe = 'Original seed=42, fraction=.2 recipe, verified by exact training fingerprint'
    if (not train or not valid or set(train) & set(valid)
            or len(train) + len(valid) != len(files) or set(train) | set(valid) != set(files)):
        raise ValueError('Archived split does not partition original images; crops or changed dataset require original manifest')
    records = [dict(id=key, source_id=key, split=split, boxes=boxes[key])
               for split, ids in (('train', train), ('valid', valid)) for key in sorted(ids)]
    info = dict(classes=list(CLASSES), source=str(dataset),
                annotation_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
                train_ids=sorted(train), valid_ids=sorted(valid), records=records)
    actual = split_fingerprint(info)
    if actual != expected:
        raise ValueError(f'Recovered split/annotations do NOT match training. Expected {expected}, got {actual}. '
                         'No manifest written. Do not compare on a newly guessed split.')
    print('Exact training fingerprint MATCHED. Reading source image sizes; images will not be copied.', flush=True)
    def get_size(key):
        with Image.open(files[key]) as image:
            return key, image.size
    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = dict(pool.map(get_size, files))
    for record in records:
        key = record['id']
        width, height = sizes[key]
        if any(x + w > width or y + h > height for _, x, y, w, h in record['boxes']):
            raise ValueError(f'Annotation exceeds image bounds: {key}')
        record.update(width=width, height=height, path=str(files[key].resolve()), crop=None)
    info.update(recovery=dict(method=recipe, training_fingerprint=expected, purpose='Ensemble evaluation metadata only'))
    output.mkdir(parents=True)
    (output / 'manifest.json').write_text(json.dumps(info), encoding='utf-8')
    print(f'YOLO split restored: train={len(train)}, validation={len(valid)}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('dataset', 'run', 'output', 'result'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--candidate', action='append', default=[], type=Path)
    args = parser.parse_args()
    folder = recover(args.dataset, args.run, args.output, args.candidate)
    args.result.write_text(json.dumps(dict(prepared=str(folder))), encoding='utf-8')
