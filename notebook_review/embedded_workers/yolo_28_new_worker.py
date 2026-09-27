"""Predict a new image folder with the existing custom YOLO26l, without a training manifest."""
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
