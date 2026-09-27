"""Standalone CPU WBF: YOLO PredictionString CSV + RF per-box xywh CSV."""
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
