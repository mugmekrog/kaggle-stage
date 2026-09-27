# %% [markdown]
# # Ensemble — metadata kurtarma ve 60/40 karşılaştırması
# 
# 03 ensemble akışının devamıdır. Hücreleri aynı Colab kernel'ine kopyalayın. ENS_*, YOLO_WEIGHTS, YOLO_CSV, RF_CSV ve ens_run tanımlı olmalı. Ortak validation verisiyle iki aday karşılaştırılır.

# %% [markdown]
# ## Metadata, split kurtarma ve karşılaştırma
# 
# Gömülü worker kaynakları ayrıca embedded_workers/ altında okunabilir.

# %%
# Run this entire cell in the EXISTING YOLO ensemble notebook.
from pathlib import Path
from datetime import datetime
import json
import subprocess
import os

needed = ('ENS_PY', 'ENS_CODE', 'ENS_DATASET', 'ENS_RESULTS', 'YOLO_WEIGHTS', 'ens_run', 'YOLO_CSV', 'RF_CSV')
assert all(k in globals() for k in needed), 'Use the existing YOLO ensemble runtime.'
META_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
META_WORKER = Path(ENS_CODE) / 'recover_yolo_manifest.py'
META_WORKER.write_text(r'''"""Recover evaluation metadata only, requiring the exact fingerprint saved during training."""
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
''', encoding='utf-8')

# %%
META_OUTPUT = Path(ENS_RESULTS) / ('ensemble_metadata_' + META_STAMP)
META_RESULT = Path(ENS_CODE) / ('metadata_path_' + META_STAMP + '.json')
META_CANDIDATES = [Path(globals()[k]) for k in ('ENS_PREPARED', 'CUSTOM_PREPARED', 'PREPARED')
                   if k in globals() and globals()[k]]
META_CANDIDATES += [Path('/content/vehicles_prepared'), Path('/content/work/vehicles_prepared'),
                    Path('/content/drive/MyDrive/vehicles_prepared')]
for p in Path('/content').glob('*/manifest.json'):
    META_CANDIDATES.append(p.parent)
meta_cmd = [str(ENS_PY), '-u', str(META_WORKER), '--dataset', str(ENS_DATASET),
            '--run', str(Path(YOLO_WEIGHTS).parent), '--output', str(META_OUTPUT), '--result', str(META_RESULT)]
for p in dict.fromkeys(META_CANDIDATES):
    meta_cmd += ['--candidate', str(p)]
print('YOLO metadata kontrol ediliyor. Eğitim yok; görüntüler kopyalanmaz.', flush=True)
meta_job = subprocess.run(meta_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          env=dict(os.environ, MPLBACKEND='Agg'))
print(meta_job.stdout[-12000:])
if meta_job.returncode:
    print(meta_job.stderr[-8000:])
    raise RuntimeError('YOLO metadata verification failed; share the output above.')
ENS_PREPARED = Path(json.loads(META_RESULT.read_text())['prepared'])
print('ENS_PREPARED:', ENS_PREPARED)

# Next: verify RF cache split and compare both 60/40 candidates.
# Paste this entire cell into the EXISTING YOLO ensemble notebook.
from pathlib import Path
import subprocess
import os

required_cache_globals = ('ENS_PY', 'ENS_CODE', 'ENS_PREPARED', 'ENS_DATASET', 'ens_run', 'YOLO_CSV', 'RF_CSV')
missing_cache_globals = [k for k in required_cache_globals if k not in globals()]
assert not missing_cache_globals, f'Use the existing YOLO ensemble runtime. Missing: {missing_cache_globals}'
RF_ROOT = Path('/content/drive/MyDrive/roketsan_astra/rfdetr_dinov2')
RF_VAL_CACHE = RF_ROOT / 'preds/rfdetr_large_768_t768__checkpoint_best_total__val.pkl'
RF_FUSION_GRID = RF_ROOT / 'runs/rfdetr_large_768_t768/fusion_grid_val.csv'
RF_SAVED_SPLIT = RF_ROOT / 'runs/rfdetr_large_768_t768/train_originals.coco.json'
assert RF_VAL_CACHE.is_file(), str(RF_VAL_CACHE)
assert RF_FUSION_GRID.is_file(), str(RF_FUSION_GRID)
RECOVERY_WORKER = Path(ENS_CODE) / 'recover_rf_split_from_cache.py'
RECOVERY_WORKER.write_text(r'''"""Recover the supplied RF notebook split, accepting only its exact cached validation IDs."""
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
''', encoding='utf-8')

# %%
recovery_job = subprocess.run(
    [str(ENS_PY), '-u', str(RECOVERY_WORKER), '--prepared', str(ENS_PREPARED),
     '--dataset', str(ENS_DATASET), '--cache', str(RF_VAL_CACHE), '--output', str(RF_SAVED_SPLIT)],
    capture_output=True, text=True, encoding='utf-8', errors='replace',
    env=dict(os.environ, MPLBACKEND='Agg'))
print(recovery_job.stdout[-12000:])
if recovery_job.returncode:
    print(recovery_job.stderr[-8000:])
    raise RuntimeError('RF split verification failed. Share the output above; no new training is needed.')

# Recovering the split succeeded. Continue directly to both 60/40 candidates.
"""Paste into a new cell AFTER the existing ensemble setup/CSV selection cells."""
from pathlib import Path
from datetime import datetime
import json
import hashlib
import shutil


def choose_6040(report):
    import math
    expected = {'ensemble_yolo60': (0.6, 0.4), 'ensemble_rf60': (0.4, 0.6)}
    candidates = [row for row in report['results'] if row['name'] in expected]
    if len(candidates) != 2 or {r['name'] for r in candidates} != set(expected):
        raise ValueError('Both 60/40 validation results are required')
    for row in candidates:
        score = float(row['mAP50'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid validation AP50')
        weights = row['weights']
        if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError('Invalid ensemble weights')
        if any(abs(w / sum(weights) - target) > 1e-9 for w, target in zip(weights, expected[row['name']])):
            raise ValueError('Validation weights do not match the named 60/40 candidate')
    # On an exact tie, prefer YOLO60 deterministically; report the tie explicitly.
    best = max(candidates, key=lambda row: (float(row['mAP50']), row['name'] == 'ensemble_yolo60'))
    return candidates, best


required = ('ens_run', 'ENS_RESULTS', 'ENS_DATASET', 'ENS_PREPARED',
            'YOLO_CSV', 'RF_CSV', 'YOLO_WEIGHTS', 'YOLO_IOU', 'YOLO_CAP', 'RF_ROOT')
missing = [key for key in required if key not in globals()]
assert not missing, f'Önce mevcut ensemble kurulum/seçim hücrelerini çalıştırın. Eksik: {missing}'
assert YOLO_WEIGHTS and Path(YOLO_WEIGHTS).is_file(), 'Doğru YOLO26l checkpointini seçin.'
def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

current_hash = file_hash(YOLO_WEIGHTS)
provenance_candidates = [Path(YOLO_CSV).parent / name for name in
                         ('submission_report.json', 'prediction_provenance.json')]
provenance = next((p for p in provenance_candidates if p.is_file()), None)
assert provenance is not None, 'YOLO test CSV checkpoint kaydı bulunamadı.'
assert json.loads(provenance.read_text()).get('checkpoint_sha256') == current_hash, 'YOLO test CSV ile checkpoint eşleşmiyor.'
PAIR_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
PAIR_ROOT = ENS_RESULTS / ('ensemble_6040_' + PAIR_STAMP)
PAIR_ROOT.mkdir()

# Reuse an already completed common-holdout comparison when available.
previous_report = Path(TUNE_OUTPUT) / 'ensemble_validation.json' if 'TUNE_OUTPUT' in globals() else None
if previous_report is not None and previous_report.is_file():
    PAIR_REPORT = json.loads(previous_report.read_text())
    # A stale report from another checkpoint must never choose the current ensemble.
    val_folders = [Path(globals()[name]) for name in ('YVAL_OUTPUT', 'YVAL') if name in globals()]
    val_folders.append(previous_report.parent.parent / 'yolo_validation')
    matching_val = [p for p in val_folders if (p / 'yolo.csv').is_file()
                    and file_hash(p / 'yolo.csv') == PAIR_REPORT['yolo_validation_sha256']]
    assert matching_val, 'Önceki validation CSV bulunamadı. TUNE_OUTPUT değişkenini silip bu hücreyi yeniden çalıştırın.'
    val_meta = json.loads((matching_val[0] / 'prediction_provenance.json').read_text())
    assert val_meta['checkpoint_sha256'] == current_hash, 'Validation başka YOLO checkpointine ait. TUNE_OUTPUT değişkenini silip yeniden çalıştırın.'
    assert val_meta['options']['iou'] == YOLO_IOU and val_meta['options']['max_det'] == YOLO_CAP, 'Validation ve test YOLO NMS ayarları farklı.'
    print('Mevcut ortak validation raporu kullanılıyor:', previous_report)
else:
    assert YOLO_WEIGHTS and Path(YOLO_WEIGHTS).is_file(), 'Doğru YOLO26l checkpointini seçin.'
    # RF and YOLO may be in different Colab runtimes. A small exported COCO file
    # containing original train IDs is sufficient; the RF images are not copied.
    local_split = Path('/content/work/rf_ds/train/_annotations.coco.json')
    saved_split = RF_ROOT / 'runs/rfdetr_large_768_t768/train_originals.coco.json'
    RF_SPLIT = local_split if local_split.is_file() else saved_split
    assert RF_SPLIT.is_file(), ('RF eğitim ayrımı bu oturumda bulunamadı. RF notebookunda '
        'export_rf_split_colab_cell.py hücresini bir kez çalıştırın, sonra bu hücreyi tekrar çalıştırın.')
    RF_CACHE = Path(globals().get('RF_VAL_CACHE', RF_ROOT / 'preds/rfdetr_large_768_t768__checkpoint_best_total__val.pkl'))
    RF_GRID = RF_ROOT / 'runs/rfdetr_large_768_t768/fusion_grid_val.csv'
    assert RF_CACHE.is_file() and RF_GRID.is_file(), f'RF validation cache/grid gerekli: {RF_CACHE}, {RF_GRID}'
    # The source notebook chooses checkpoint_best_total for its final CSV when available.
    # If that notebook was changed to EMA, set RF_VAL_CACHE to its matching EMA cache.
    HOLDOUT = PAIR_ROOT / 'holdout'
    ens_run('holdout', '--data', ENS_PREPARED, '--rf-train-coco', RF_SPLIT, '--output', HOLDOUT)
    IDS = HOLDOUT / 'common_holdout.json'
    YVAL = PAIR_ROOT / 'yolo_validation'
    ens_run('predict-yolo', '--weights', YOLO_WEIGHTS, '--dataset', ENS_DATASET, '--ids', IDS,
            '--iou', YOLO_IOU, '--max-det', YOLO_CAP, '--output', YVAL)
    RVAL = PAIR_ROOT / 'rf_validation'
    ens_run('rf-cache', '--ids', IDS, '--cache', RF_CACHE, '--fusion-grid', RF_GRID, '--output', RVAL)
    TUNE_OUTPUT = PAIR_ROOT / 'validation_comparison'
    ens_run('tune', '--data', ENS_PREPARED, '--rf-train-coco', RF_SPLIT,
            '--yolo', YVAL / 'yolo.csv', '--rf', RVAL / 'rf_validation.csv', '--output', TUNE_OUTPUT)
    PAIR_REPORT = json.loads((TUNE_OUTPUT / 'ensemble_validation.json').read_text())

# %%
assert PAIR_REPORT['common_images'] > 0, 'Ortak validation sonucu gerekli.'
candidates, best = choose_6040(PAIR_REPORT)
paths = {}
for row in candidates:
    choice = dict(PAIR_REPORT, selected=row['name'], weights=row['weights'])
    choice_file = PAIR_ROOT / (row['name'] + '_selection.json')
    choice_file.write_text(json.dumps(choice, indent=2), encoding='utf-8')
    destination = PAIR_ROOT / row['name']
    ens_run('submit', '--yolo', YOLO_CSV, '--rf', RF_CSV, '--dataset', ENS_DATASET,
            '--selection', choice_file, '--output', destination)
    paths[row['name']] = destination / 'submission.csv'

# The winner is chosen solely from shared-validation AP50, never from test/public scores.
WINNER_CSV = PAIR_ROOT / 'submission.csv'
shutil.copyfile(paths[best['name']], WINNER_CSV)
result = dict(common_images=PAIR_REPORT['common_images'],
              candidates=[dict(name=r['name'], weights=r['weights'], mAP50=r['mAP50'],
                               per_class=r['per_class'], csv=str(paths[r['name']])) for r in candidates],
              selected=best['name'], selected_validation_AP50=best['mAP50'],
              tie=candidates[0]['mAP50'] == candidates[1]['mAP50'],
              overall_best_reference=max(PAIR_REPORT['results'], key=lambda row: row['mAP50'])['name'],
              test_mAP=None, final_submission=str(WINNER_CSV))
(PAIR_ROOT / 'comparison_6040.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
for row in candidates:
    print(row['name'], f"validation mAP50: {row['mAP50']:.6f}")
print('SEÇİLEN:', best['name'])
print('YARIŞMAYA GÖNDER:', WINNER_CSV)
