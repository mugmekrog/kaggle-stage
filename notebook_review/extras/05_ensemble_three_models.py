# %% [markdown]
# # ConvNeXt + YOLO + RF-DETR — sabit ağırlıklı ensemble
# 
# 03 ensemble akışının devamıdır; aynı kernel gereklidir. ConvNeXt CSV yolunu kontrol edin. Ağırlıklar %50 / %25 / %25; validation tuning yapılmaz.

# %% [markdown]
# ## Üç model fusion
# 
# Gömülü worker kaynakları ayrıca embedded_workers/ altında okunabilir.

# %%
# Paste into a new cell in the EXISTING YOLO ensemble Colab runtime.
from pathlib import Path
from datetime import datetime
from collections import deque
import subprocess
import os
import time
from IPython.display import clear_output

required_three = ('ENS_PY', 'ENS_CODE', 'ENS_DATASET', 'ENS_RESULTS', 'YOLO_CSV', 'RF_CSV')
missing_three = [k for k in required_three if k not in globals()]
assert not missing_three, f'Use the existing ensemble notebook. Missing: {missing_three}'

# User-provided path first. Notebook's alternate folder is only a fallback.
if 'CONVNEXT_CSV' not in globals():
    CONVNEXT_CSV = Path('/content/drive/MyDrive/roketsan_runs/cnv2b_f0/submission_final.csv')
    if not CONVNEXT_CSV.is_file():
        alternate = Path('/content/drive/MyDrive/roketsan_runs/convnext_cascade/cnv2b_f0/submission_final.csv')
        if alternate.is_file():
            CONVNEXT_CSV = alternate
CONVNEXT_CSV = Path(CONVNEXT_CSV)
for name, p in [('ConvNeXt', CONVNEXT_CSV), ('YOLO26l', Path(YOLO_CSV)), ('RF-DETR', Path(RF_CSV))]:
    assert p.is_file(), f'{name} CSV missing: {p}'
    print(name, p)
assert (Path(ENS_CODE) / 'ensemble_vehicle.py').is_file(), 'Existing ensemble code is missing.'
THREE_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
THREE_OUTPUT = Path(ENS_RESULTS) / ('ensemble_conv50_yolo25_rf25_' + THREE_STAMP)
THREE_WORKER = Path(ENS_CODE) / 'ensemble_three.py'
THREE_WORKER.write_text(r'''"""Fixed 50% ConvNeXt / 25% YOLO / 25% RF-DETR test candidate from original model CSVs."""
import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from common.runtime import CLASSES, write_json
from common.predictions import sample_ids, image_files, parse_prediction_string
from ensemble_vehicle import read_predictions, checksum, fuse, serialize
from validate_submission import validate


def source_stats(predictions):
    counts = Counter()
    scores = []
    small = 0
    for rows in predictions.values():
        for c, score, x1, y1, x2, y2 in rows:
            counts[CLASSES[c]] += 1
            scores.append(score)
            small += (x2 - x1) * (y2 - y1) < 200
    return dict(images=len(predictions), boxes=len(scores),
                empty_images=sum(not rows for rows in predictions.values()),
                per_class=dict((c, counts[c]) for c in CLASSES),
                boxes_below_200=int(small),
                confidence_quantiles=dict(zip(('p05', 'p50', 'p95'),
                    map(float, np.quantile(scores, [.05, .5, .95])))) if scores else None)


def run(dataset, convnext, yolo, rf, output):
    dataset, output = Path(dataset), Path(output)
    paths = list(map(Path, (convnext, yolo, rf)))
    names = ['convnext', 'yolo26l', 'rfdetr']
    weights = [2.0, 1.0, 1.0]  # normalized: .50 / .25 / .25
    ids = sample_ids(dataset / 'sample_submission.csv')
    images = image_files(dataset / 'test/images')
    if set(images) != set(ids):
        raise ValueError('Sample submission and test image IDs differ')
    if output.exists():
        raise FileExistsError('Choose a new output directory; previous submissions are preserved')
    for path in paths:
        if (path.parent / 'ensemble_report.json').is_file() or (path.parent / 'comparison_6040.json').is_file():
            raise ValueError(f'Use each original model CSV, not an already fused ensemble: {path}')
    if len({p.resolve() for p in paths}) != 3:
        raise ValueError('Three distinct source CSVs are required')
    hashes = [checksum(p) for p in paths]
    predictions, statistics = [], {}
    for name, path in zip(names, paths):
        print('Reading:', name, path, flush=True)
        rows = read_predictions(path)
        if set(rows) != set(ids):
            raise ValueError(f'{name}: CSV must contain exactly all sample IDs; missing={len(set(ids)-set(rows))}, '
                             f'extra={len(set(rows)-set(ids))}')
        predictions.append(rows)
        statistics[name] = source_stats(rows)
    output.mkdir(parents=True)
    candidate = output / 'submission_candidate.csv'
    out_counts, empty, detections = Counter(), 0, 0
    with candidate.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_id', 'PredictionString'])
        for index, key in enumerate(ids, 1):
            with Image.open(images[key]) as im:
                width, height = im.size
            boxes = fuse([p[key] for p in predictions], width, height,
                         weights=weights, iou=.55, confidence_type='avg')
            value = serialize(boxes, width, height)
            writer.writerow([key, value])
            kept = parse_prediction_string(value)
            empty += not kept
            detections += len(kept)
            out_counts.update(CLASSES[b[0]] for b in kept)
            if index % 100 == 0 or index == len(ids):
                print(f'Three-model fusion: {index}/{len(ids)}', flush=True)
    verification = validate(dataset, candidate)
    if [checksum(p) for p in paths] != hashes:
        raise RuntimeError('An input CSV changed during fusion; final submission not published')
    final = output / 'submission.csv'
    candidate.replace(final)
    report = dict(candidate='convnext50_yolo25_rf25',
                  weights=dict(zip(names, [.5, .25, .25])),
                  inputs={n: dict(path=str(p), sha256=h, statistics=statistics[n])
                          for n, p, h in zip(names, paths, hashes)},
                  fusion=dict(method='class-wise weighted boxes fusion', iou=.55,
                              confidence_type='avg', skip_box_threshold=.001, final_minimum_area=200),
                  output=dict(path=str(final), sha256=checksum(final), detections=detections,
                              empty_images=empty, per_class=dict(out_counts)),
                  verification=verification, validation_tuned=False,
                  validation_mAP50=None, public_mAP50=None,
                  note='Fixed-weight test candidate. Confidence calibration and joint held-out performance '
                       'have not been measured. No checkpoint loaded or inference performed.')
    write_json(output / 'ensemble_report.json', report)
    print('FINAL SUBMISSION:', final, flush=True)
    print('CASE REPORT:', output / 'ensemble_report.json', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'convnext', 'yolo', 'rf', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    run(**vars(parser.parse_args()))
''', encoding='utf-8')

# %%
THREE_LOG = Path(ENS_RESULTS) / '_logs' / ('ensemble_three_' + THREE_STAMP + '.log')
THREE_LOG.parent.mkdir(parents=True, exist_ok=True)
three_cmd = [str(ENS_PY), '-u', str(THREE_WORKER), '--dataset', str(ENS_DATASET),
             '--convnext', str(CONVNEXT_CSV), '--yolo', str(YOLO_CSV), '--rf', str(RF_CSV),
             '--output', str(THREE_OUTPUT)]
three_tail, three_last = deque(maxlen=20), time.monotonic()
print('CPU fusion: 50% ConvNeXt / 25% YOLO26l / 25% RF-DETR. No training or inference.')
print('Log:', THREE_LOG, flush=True)
with THREE_LOG.open('w', encoding='utf-8') as log:
    log.write(repr(three_cmd) + '\n')
    with subprocess.Popen(three_cmd, cwd=ENS_CODE, env=dict(os.environ, MPLBACKEND='Agg'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as p:
        try:
            for line in p.stdout:
                log.write(line); log.flush()
                if line.strip(): three_tail.append(line)
                if time.monotonic() - three_last > 20:
                    clear_output(wait=True)
                    print('Log:', THREE_LOG)
                    print(''.join(three_tail)[-10000:])
                    three_last = time.monotonic()
            three_exit = p.wait()
        except BaseException:
            p.terminate()
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
            raise
clear_output(wait=True)
print('Log:', THREE_LOG)
print(''.join(three_tail)[-10000:])
if three_exit:
    raise RuntimeError(f'Three-model fusion failed. Share the last error above. Full log: {THREE_LOG}')
print('SUBMISSION:', THREE_OUTPUT / 'submission.csv')
print('REPORT:', THREE_OUTPUT / 'ensemble_report.json')
print('Fixed-weight candidate; validation tuning was not performed.')
