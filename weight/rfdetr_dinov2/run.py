"""RF-DETR 1.4.0 with DINOv2: fine-tune pretrained detector and predict."""
from __future__ import annotations

import os
import importlib.metadata
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from common.runtime import (CLASSES, checkpoint_metadata, new_run, parser_for,
                            validate_train, write_json)
from common.predictions import run_prediction


def build_model(variant, resolution, device, weights=None, **kwargs):
    from rfdetr import RFDETRSmall, RFDETRMedium
    from rfdetr.main import download_pretrain_weights
    kind = {'small':RFDETRSmall,'medium':RFDETRMedium}[variant]
    if resolution % 32:
        raise ValueError('These RF-DETR variants require resolution divisible by 32')
    # This pinned release normalizes pretrain paths before its download lookup.
    # Explicitly download the published named checkpoint first, then pass its path.
    if weights is None:
        filename = f'rf-detr-{variant}.pth'
        cache = Path(__file__).resolve().parents[1]/'checkpoints'
        cache.mkdir(exist_ok=True)
        previous = Path.cwd()
        try:
            os.chdir(cache)
            download_pretrain_weights(filename)
        finally:
            os.chdir(previous)
        weights = cache/filename
    if device.startswith('cuda:'):
        import torch
        torch.cuda.set_device(int(device.split(':')[1]))
        device = 'cuda'
    return kind(pretrain_weights=str(Path(weights).resolve()), resolution=resolution,
                device=device, **kwargs)


def train(args):
    validate_train(args)
    if args.imgsz % 32 or args.grad_accum < 1 or not 1 <= args.groups <= 13 or args.lr <= 0:
        raise ValueError('Use resolution divisible by 32, positive accumulation and 1..13 query groups')
    if args.dry_run:
        return
    import torch
    version = importlib.metadata.version('rfdetr')
    if version != '1.4.0':
        raise RuntimeError('Use rfdetr==1.4.0: this adapter relies on its category and callback APIs')
    out = new_run(args,'rfdetr_dinov2')
    model = build_model(args.model,args.imgsz,args.device,group_detr=args.groups,
                        gradient_checkpointing=True)
    best = -1.0

    def save_ap50(log):
        nonlocal best
        candidates = []
        for key, network in [('test_coco_eval_bbox',model.model.model),
                             ('ema_test_coco_eval_bbox',getattr(getattr(model.model,'ema_m',None),'module',None))]:
            if network is not None and key in log:
                candidates.append((float(log[key][1]),network,key))
        if not candidates:
            raise RuntimeError('RF-DETR validation AP50 missing from callback; check pinned dependency')
        score,network,key = max(candidates,key=lambda row:row[0])
        if score > best:
            best = score
            torch.save({'model':{k:v.detach().cpu() for k,v in network.state_dict().items()},
                        'args':model.model.args,'epoch':log['epoch']},out/'best_ap50.pth')
            write_json(out/'best_ap50.json',{'epoch':log['epoch']+1,'validation_AP50':score,'branch':key})

    model.callbacks['on_fit_epoch_end'].append(save_ap50)
    model.train(dataset_dir=str(args.data.resolve()/'coco'),dataset_file='roboflow',
                output_dir=str(out),epochs=args.epochs,batch_size=args.batch,
                grad_accum_steps=args.grad_accum,num_workers=args.workers,
                lr=args.lr,lr_encoder=args.lr*0.1,seed=args.seed,
                group_detr=args.groups,multi_scale=False,expanded_scales=False,
                run_test=False,eval_max_dets=500,tensorboard=False,wandb=False,
                early_stopping=False,checkpoint_interval=10,
                lr_drop=max(1,int(args.epochs*0.8)),class_names=list(CLASSES))
    print(f'AP50-selected checkpoint: {out / "best_ap50.pth"}')


def predict(args):
    info = checkpoint_metadata(args.weights,'rfdetr_dinov2')
    if importlib.metadata.version('rfdetr') != '1.4.0':
        raise RuntimeError('Use rfdetr==1.4.0 for the saved class-ID convention')
    model = build_model(info['model'],info['imgsz'],args.device,args.weights,
                        num_classes=4,group_detr=info['groups'])
    if model.class_names != {i+1:name for i,name in enumerate(CLASSES)}:
        raise ValueError('Checkpoint is missing the expected one-based RF-DETR class names')
    def predictor(image):
        result = model.predict(image,threshold=args.confidence)
        detections = []
        for c,s,b in zip(result.class_id,result.confidence,result.xyxy):
            if int(c) == 0:
                continue  # Reserved unused category slot in this one-based COCO export.
            if int(c) not in (1,2,3,4):
                raise ValueError(f'Unexpected RF-DETR category ID: {c}')
            detections.append((int(c)-1,float(s),*map(float,b)))
        return detections
    run_prediction(args,predictor)


def main():
    parser, training, _ = parser_for('rfdetr_dinov2',768,1)
    training.add_argument('--model',choices=('small','medium'),default='medium')
    training.add_argument('--grad-accum',type=int,default=8)
    training.add_argument('--groups',type=int,default=13,help='Use 1 to reduce training memory; default pretrained recipe uses 13')
    training.add_argument('--lr',type=float,default=1e-4)
    args = parser.parse_args()
    (train if args.command=='train' else predict)(args)


if __name__=='__main__':
    main()
