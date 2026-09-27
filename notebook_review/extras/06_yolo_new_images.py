# %% [markdown]
# # YOLO — yeni görüntüler ve RF ile 50/50 ensemble
# 
# YOLO kurulum hücreleri aynı kernel'de çalışmış olmalı; bu hücreleri o oturuma kopyalayın. Varsayılan klasör roketsan_dataset/images, beklenen görüntü sayısı 40. İlk iki bölüm yalnız YOLO tahmini üretir; son bölüm RF boxes.csv gerektiren isteğe bağlı ensemble'dır. Run all kullanmayın; son bölümü yalnız iki model çıktısı hazırsa çalıştırın.

# %% [markdown]
# ## Checkpoint seçimi
# 
# Gömülü worker kaynakları ayrıca embedded_workers/ altında okunabilir.

# %%
from pathlib import Path
import json

RESULTS = Path("/content/drive/MyDrive/roketsan_results")

def validation_score(weights):
    report = weights.parent / "best_ap50.json"
    if report.is_file():
        return float(json.loads(report.read_text())["validation_AP50"])
    return -1.0

candidates = []
for weights in RESULTS.glob("yolo26l_*/best_ap50.pt"):
    if any(tag in weights.parent.name.lower() for tag in ("smoke", "audit")):
        continue
    metadata = weights.parent / "run.json"
    if metadata.is_file():
        info = json.loads(metadata.read_text())
        if info.get("model") == "l":
            candidates.append(weights)

assert candidates, (
    f"Checkpoint bulunamadı: {RESULTS}. "
    "Drive bağlı mı ve best_ap50.pt hangi klasörde kontrol et."
)

candidates.sort(key=validation_score, reverse=True)

if len(candidates) > 1 and validation_score(candidates[0]) < 0:
    print(*candidates, sep="\n")
    raise RuntimeError("Skor kaydı yok; yukarıdaki checkpoint yollarını paylaş.")

NEW_WEIGHTS = candidates[0]
print("Seçilen checkpoint:", NEW_WEIGHTS)
print("Kayıtlı validation AP50:", validation_score(NEW_WEIGHTS))

# %% [markdown]
# ## Yeni görüntü tahmini
# 
# Gömülü worker kaynakları ayrıca embedded_workers/ altında okunabilir.

# %%
# Paste into a new cell in the existing YOLO26l Colab notebook.
from pathlib import Path
from datetime import datetime
import subprocess
import os

NEW_IMAGES = Path('/content/drive/MyDrive/roketsan_dataset/images')
EXPECTED_NEW_IMAGES = 40
assert NEW_IMAGES.is_dir(), f'Image folder not found: {NEW_IMAGES}'

def first_existing(names, directory=False):
    for name in names:
        value = globals().get(name)
        if value:
            path = Path(value)
            if (path.is_dir() if directory else path.is_file()): return path
    return None

NEW_PY = first_existing(('CUSTOM_PY', 'ENS_PY', 'PY'))
NEW_CODE = next((Path(globals()[k]) for k in ('CUSTOM_PROJECT', 'ENS_CODE', 'PROJECT')
                 if globals().get(k) and (Path(globals()[k]) / 'yolo26l_custom/infer.py').is_file()), None)
NEW_WEIGHTS = first_existing(('NEW_WEIGHTS', 'CUSTOM_WEIGHTS', 'YOLO_WEIGHTS'))
if NEW_WEIGHTS is None and globals().get('CUSTOM_RUN'):
    NEW_WEIGHTS = Path(CUSTOM_RUN) / 'best_ap50.pt'
assert NEW_PY and NEW_CODE, 'Use the existing YOLO26l notebook/runtime; code or environment is missing.'
assert NEW_WEIGHTS and NEW_WEIGHTS.is_file(), 'Set NEW_WEIGHTS = Path("exact/best_ap50.pt") first.'
NEW_OUTPUT = Path('/content/drive/MyDrive/roketsan_results') / ('yolo26l_new40_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
NEW_WORKER = NEW_CODE / 'predict_yolo_folder.py'
NEW_WORKER.write_text(r'''"""Predict a new image folder with the existing custom YOLO26l, without a training manifest."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from PIL import Image
from common.runtime import CLASSES, checkpoint_metadata, write_json
from common.predictions import image_files, parse_prediction_string


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export_csv(raw, images, output):
    ids = sorted(images)
    with Path(raw).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['image_id', 'PredictionString']:
            raise ValueError('Invalid raw CSV header')
        rows = list(reader)
    if [row['image_id'] for row in rows] != ids:
        raise ValueError('Inference CSV IDs differ from input images')
    removed, kept, empty = 0, 0, 0
    with Path(output).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_id', 'PredictionString'])
        for row in rows:
            detections = parse_prediction_string(row['PredictionString'])
            with Image.open(images[row['image_id']]) as im:
                width, height = im.size
            if any(b[4] > width + .001 or b[5] > height + .001 for b in detections):
                raise ValueError('Predicted box exceeds original image bounds')
            tokens = row['PredictionString'].split() if detections else []
            groups = []
            for offset in range(0, len(tokens), 6):
                group = tokens[offset:offset + 6]
                if Decimal(group[4]) * Decimal(group[5]) >= Decimal(200):
                    groups.extend(group)
                    kept += 1
                else:
                    removed += 1
            empty += not groups
            writer.writerow([row['image_id'], ' '.join(groups) or 'none'])
    return dict(images=len(ids), detections=kept, empty_images=empty, removed_below_200=removed)


def run(args):
    from yolo26l_custom.run import library_check
    from yolo26l_custom.infer import infer_images
    environment = library_check()
    if not environment['cuda_available']:
        raise RuntimeError('Enable a GPU runtime in the existing YOLO notebook')
    meta = checkpoint_metadata(args.weights, 'yolo26')
    if meta.get('model') != 'l':
        raise ValueError('Select the YOLO26l checkpoint')
    images = image_files(args.images)
    if len(images) != args.expected_images:
        raise ValueError(f'Expected {args.expected_images} images, found {len(images)} JPEG/PNG files in {args.images}')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    weights_hash = digest(args.weights)
    options = dict(imgsz=int(meta['imgsz']), conf=.001, iou=.7, max_det=500,
                   nms=True, agnostic_nms=False, rect=True, augment=False, quantize=32,
                   device='0', verbose=False)
    selection_path = None
    for path in sorted(args.weights.parent.glob('nms_validation_*/nms_selection.json'), reverse=True):
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('checkpoint_sha256') == weights_hash:
            options.update(saved['options'])
            options.update(iou=saved['selected']['iou'], max_det=saved['selected']['max_det'],
                           device='0', verbose=False)
            selection_path = str(path)
            break
    print('Weights:', args.weights, flush=True)
    print('Images:', len(images), '| NMS selection:', selection_path or 'default .7 / 500', flush=True)
    print('Inference settings:', options, flush=True)
    args.output.mkdir(parents=True)
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    if [model.names[i] for i in range(len(model.names))] != list(CLASSES):
        raise ValueError('Unexpected checkpoint class mapping')
    paths = infer_images(model, images, sorted(images),
                         {'raw_predictions': (options['iou'], options['max_det'])}, options, args.output)
    candidate = args.output / 'predictions_candidate.csv'
    verification = export_csv(paths['raw_predictions'], images, candidate)
    if digest(args.weights) != weights_hash:
        raise RuntimeError('Checkpoint changed during inference')
    final = args.output / 'predictions.csv'
    candidate.replace(final)
    write_json(args.output / 'prediction_report.json', dict(weights=str(args.weights),
        checkpoint_sha256=weights_hash, source_folder=str(args.images), ids=sorted(images),
        options=options, nms_selection=selection_path, minimum_area=200,
        verification=verification, output=str(final), mAP50=None,
        note='New image-folder predictions; separate from the original competition sample submission.'))
    print('CSV:', final, flush=True)
    print('CHECK:', verification, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'weights', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--expected-images', type=int, default=40)
    run(p.parse_args())
''', encoding='utf-8')

# %%
print('YOLO26l checkpoint:', NEW_WEIGHTS)
print('New images:', NEW_IMAGES)
print('40 görüntü için GPU inference başlıyor. Yeniden eğitim yapılmaz.', flush=True)
new_job = subprocess.run([str(NEW_PY), '-u', str(NEW_WORKER), '--images', str(NEW_IMAGES),
    '--weights', str(NEW_WEIGHTS), '--output', str(NEW_OUTPUT), '--expected-images', str(EXPECTED_NEW_IMAGES)],
    cwd=NEW_CODE, env=dict(os.environ, MPLBACKEND='Agg'), capture_output=True,
    text=True, encoding='utf-8', errors='replace')
NEW_LOG = NEW_OUTPUT.parent / (NEW_OUTPUT.name + '.log')
NEW_LOG.parent.mkdir(parents=True, exist_ok=True)
NEW_LOG.write_text(new_job.stdout + '\n' + new_job.stderr, encoding='utf-8')
print(new_job.stdout[-10000:])
if new_job.returncode:
    print(new_job.stderr[-8000:])
    raise RuntimeError(f'Prediction failed. Full log: {NEW_LOG}')
print('CSV HAZIR:', NEW_OUTPUT / 'predictions.csv')
print('RAPOR:', NEW_OUTPUT / 'prediction_report.json')

# %% [markdown]
# ## İsteğe bağlı RF + YOLO fusion
# 
# Gömülü worker kaynakları ayrıca embedded_workers/ altında okunabilir.

# %%
# Paste into the notebook where the NEW 40-image YOLO CSV was produced.
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os

PAIR40_IMAGES = Path('/content/drive/MyDrive/roketsan_dataset/images')
PAIR40_RF = Path('/content/drive/MyDrive/roketsan_astra/rfdetr_dinov2/boxes.csv')
if 'PAIR40_YOLO' not in globals():
    if globals().get('NEW_OUTPUT') and (Path(NEW_OUTPUT) / 'predictions.csv').is_file():
        PAIR40_YOLO = Path(NEW_OUTPUT) / 'predictions.csv'
    else:
        matches = sorted(Path('/content/drive/MyDrive/roketsan_results').glob('yolo26l_new40_*/predictions.csv'))
        if len(matches) != 1:
            print('New YOLO CSV candidates:', *map(str, matches), sep='\n')
            raise RuntimeError('Set PAIR40_YOLO = Path("exact/new40/predictions.csv") above this cell.')
        PAIR40_YOLO = matches[0]
PAIR40_YOLO = Path(PAIR40_YOLO)
for path in (PAIR40_YOLO, PAIR40_RF):
    assert path.is_file(), f'CSV not found: {path}'
assert PAIR40_IMAGES.is_dir(), str(PAIR40_IMAGES)

# Reuse the current Python environment; model code and checkpoints are unnecessary.
PAIR40_PY = next((str(globals()[k]) for k in ('NEW_PY', 'CUSTOM_PY', 'ENS_PY', 'PY')
                 if globals().get(k) and Path(globals()[k]).is_file()), sys.executable)
probe = subprocess.run([PAIR40_PY, '-c', 'import ensemble_boxes; from PIL import Image'], capture_output=True)
if probe.returncode:
    print('Installing CPU fusion dependencies only...', flush=True)
    setup = subprocess.run([PAIR40_PY, '-m', 'pip', 'install', 'ensemble-boxes==1.0.9', 'pillow'],
                           capture_output=True, text=True)
    if setup.returncode:
        print((setup.stdout + setup.stderr)[-8000:])
        raise RuntimeError('CPU fusion dependency installation failed.')
PAIR40_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
PAIR40_OUTPUT = Path('/content/drive/MyDrive/roketsan_results') / ('ensemble_new40_5050_' + PAIR40_STAMP)
PAIR40_WORKER = Path('/content') / ('ensemble_new40_' + PAIR40_STAMP + '.py')
PAIR40_WORKER.write_text(r'''"""Standalone CPU WBF: YOLO PredictionString CSV + RF per-box xywh CSV."""
import argparse
import csv
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from PIL import Image

CLASSES = ('car', 'van', 'truck', 'bus')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def box(label, confidence, x, y, w, h):
    if label not in CLASSES:
        raise ValueError(f'Invalid class: {label}')
    score, x, y, w, h = map(float, (confidence, x, y, w, h))
    if (not all(map(math.isfinite, (score, x, y, w, h))) or not 0 <= score <= 1
            or min(x, y) < 0 or min(w, h) <= 0):
        raise ValueError('Invalid confidence or pixel xywh coordinates')
    return (CLASSES.index(label), score, x, y, x + w, y + h)


def read_yolo(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['image_id', 'PredictionString']:
            raise ValueError('YOLO CSV must have image_id,PredictionString columns')
        for row in reader:
            key, value = row['image_id'], row['PredictionString']
            if not key or key in result:
                raise ValueError('Empty or duplicate YOLO image ID')
            if value == 'none':
                result[key] = []
                continue
            tokens = value.split()
            if not tokens or len(tokens) % 6:
                raise ValueError('YOLO PredictionString must be groups of six or none')
            result[key] = [box(*tokens[i:i + 6]) for i in range(0, len(tokens), 6)]
    return result


def read_rf(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected = ['image_id', 'label', 'confidence', 'x', 'y', 'w', 'h']
        if reader.fieldnames != expected:
            raise ValueError(f'RF CSV must have these columns: {expected}')
        for row in reader:
            if not row['image_id']:
                raise ValueError('Empty RF image ID')
            result.setdefault(row['image_id'], []).append(box(*(row[k] for k in expected[1:])))
    return result


def serialize(boxes, scores, labels, width, height):
    groups, classes, small = [], Counter(), 0
    for b, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(max(0, min(1, float(v))) * side, 3)
                          for v, side in zip(b, (width, height, width, height))]
        coords = [f'{v:.3f}' for v in (x1, y1, x2 - x1, y2 - y1)]
        if x2 <= x1 or y2 <= y1 or Decimal(coords[2]) * Decimal(coords[3]) < Decimal(200):
            small += 1
            continue
        label = CLASSES[int(label)]
        groups.append(f'{label} {float(score):.8f} ' + ' '.join(coords))
        classes[label] += 1
    return ' '.join(groups) or 'none', classes, small


def run(images, yolo, rf, output, expected_images=40):
    from ensemble_boxes import weighted_boxes_fusion
    images, output = Path(images), Path(output)
    paths = {}
    for path in sorted(images.iterdir()):
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            if path.stem in paths:
                raise ValueError('Duplicate image stem in input folder')
            paths[path.stem] = path
    if len(paths) != expected_images:
        raise ValueError(f'Expected {expected_images} images; found {len(paths)}')
    if output.exists():
        raise FileExistsError('Output exists; use a fresh folder')
    before = [digest(p) for p in (yolo, rf)]
    predictions = [read_yolo(yolo), read_rf(rf)]
    for name, rows in zip(('YOLO', 'RF'), predictions):
        if set(rows) != set(paths):
            raise ValueError(f'{name} IDs do not match the image folder. '
                f'Missing: {sorted(set(paths) - set(rows))}; extra: {sorted(set(rows) - set(paths))}. '
                'Missing RF rows are not silently treated as empty predictions.')
    print('Both CSVs match all', len(paths), 'images.', flush=True)
    print('Input boxes:', [sum(map(len, p.values())) for p in predictions], flush=True)
    output.mkdir(parents=True)
    candidate = output / 'submission_candidate.csv'
    counts, empty, removed, clipped = Counter(), 0, 0, 0
    ids = list(predictions[0])  # preserve the new YOLO CSV's order
    with candidate.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_id', 'PredictionString'])
        for index, key in enumerate(ids, 1):
            with Image.open(paths[key]) as im:
                width, height = im.size
            B, S, L = [], [], []
            for rows in predictions:
                bb, ss, ll = [], [], []
                for c, s, x1, y1, x2, y2 in rows[key]:
                    if x1 >= width or y1 >= height or x2 > width + 1.001 or y2 > height + 1.001:
                        raise ValueError(f'Box exceeds pixel bounds: {key}; check RF coordinate system')
                    clipped += x2 > width or y2 > height
                    bb.append([x1 / width, y1 / height, min(x2, width) / width, min(y2, height) / height])
                    ss.append(s); ll.append(c)
                B.append(bb); S.append(ss); L.append(ll)
            if any(S):
                fused, scores, labels = weighted_boxes_fusion(B, S, L, weights=[1, 1],
                    iou_thr=.55, skip_box_thr=.001, conf_type='avg')
            else:
                fused, scores, labels = [], [], []
            value, current, small = serialize(fused, scores, labels, width, height)
            counts.update(current); removed += small; empty += value == 'none'
            writer.writerow([key, value])
            if index % 10 == 0 or index == len(ids):
                print(f'Fusion: {index}/{len(ids)}', flush=True)
    # Validate exactly what was serialized before publishing the final file.
    check = read_yolo(candidate)
    if list(check) != ids or any((b[4]-b[2])*(b[5]-b[3]) < 200 - 1e-7 for r in check.values() for b in r):
        raise ValueError('Final CSV validation failed')
    if before != [digest(p) for p in (yolo, rf)]:
        raise RuntimeError('A source CSV changed during fusion')
    final = output / 'submission.csv'
    candidate.replace(final)
    report = dict(images=len(ids), ids=ids, weights=dict(yolo=.5, rfdetr=.5),
        sources=dict(yolo=dict(path=str(yolo), sha256=before[0]), rf=dict(path=str(rf), sha256=before[1])),
        detections=sum(counts.values()), per_class=dict(counts), empty_images=empty,
        removed_small_fused_boxes=removed, rounding_boundary_clips=clipped,
        settings=dict(iou=.55, minimum_area=200, confidence_type='avg', skip_box_threshold=.001),
        validation_tuned=False, mAP50=None, output=str(final), output_sha256=digest(final))
    (output / 'ensemble_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('CSV READY:', final, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'yolo', 'rf', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-images', type=int, default=40)
    run(**vars(parser.parse_args()))
''', encoding='utf-8')

# %%
print('YOLO:', PAIR40_YOLO)
print('RF:', PAIR40_RF)
print('40-image CPU ensemble: 50% YOLO + 50% RF-DETR.', flush=True)
pair40_job = subprocess.run([PAIR40_PY, '-u', str(PAIR40_WORKER), '--images', str(PAIR40_IMAGES),
    '--yolo', str(PAIR40_YOLO), '--rf', str(PAIR40_RF), '--output', str(PAIR40_OUTPUT)],
    capture_output=True, text=True, encoding='utf-8', errors='replace')
PAIR40_LOG = PAIR40_OUTPUT.parent / (PAIR40_OUTPUT.name + '.log')
PAIR40_LOG.parent.mkdir(parents=True, exist_ok=True)
PAIR40_LOG.write_text(pair40_job.stdout + '\n' + pair40_job.stderr, encoding='utf-8')
print(pair40_job.stdout[-10000:])
if pair40_job.returncode:
    print(pair40_job.stderr[-8000:])
    raise RuntimeError(f'Fusion failed; share the error above. Full log: {PAIR40_LOG}')
print('CSV HAZIR:', PAIR40_OUTPUT / 'submission.csv')
print('RAPOR:', PAIR40_OUTPUT / 'ensemble_report.json')
# To download afterwards:
# from google.colab import files
# files.download(str(PAIR40_OUTPUT / 'submission.csv'))
