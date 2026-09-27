"""Fixed 50% ConvNeXt / 25% YOLO / 25% RF-DETR test candidate from original model CSVs."""
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
