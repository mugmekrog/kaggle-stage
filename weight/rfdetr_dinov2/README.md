# RF-DETR Medium with DINOv2

Fine-tunes a released full detection checkpoint. The model is not a generic
DINOv2 backbone with a randomly attached decoder. This adapter explicitly targets
RF-DETR 1.4.0; its category IDs and callback API are version-sensitive.

## Install (Python 3.11 recommended)

Cloud Linux host:

```bash
python -m venv .venv-rfdetr
source .venv-rfdetr/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r rfdetr_dinov2/requirements.txt
```

On Windows, activate with `.venv-rfdetr\Scripts\Activate.ps1`. Confirm CUDA is available.
Downloads are stored in `checkpoints/`. The runtime/model dependencies have not
been installed or GPU-tested in the current workspace.

## Train

Optional smaller-model debugging configuration (not the main cloud experiment):

```bash
python rfdetr_dinov2/run.py train --model small --imgsz 576 --batch 1 --grad-accum 8 --groups 1 --epochs 1 --output runs/rfdetr_smoke
```

Main comparison run (more memory):

```bash
python rfdetr_dinov2/run.py train --model medium --imgsz 768 --batch 1 --grad-accum 8 --epochs 80 --output runs/rfdetr_dinov2
```

Resolution must be divisible by 32 for these variants. The backbone uses gradient
checkpointing and a learning rate one tenth of the main rate. Multiscale training
is disabled for predictable peak memory. `--groups 1` reduces training query groups
from the default 13; it is a different training recipe, not a guaranteed accuracy
equivalent. Keep the same groups value when reloading (restored from `run.json`).

An epoch-end callback compares the regular and EMA AP50 values and saves the
winning state as `best_ap50.pth`, independently of the package's usual AP50:95
checkpoint selection. No actual competition test evaluation is performed.

## Predict

```bash
python rfdetr_dinov2/run.py predict --weights runs/rfdetr_dinov2/best_ap50.pth --split valid --output runs/rfdetr_dinov2/valid.csv
python evaluate.py --predictions runs/rfdetr_dinov2/valid.csv
python rfdetr_dinov2/run.py predict --weights runs/rfdetr_dinov2/best_ap50.pth --tile-size 768 --output runs/rfdetr_dinov2/submission.csv
python validate_submission.py --submission runs/rfdetr_dinov2/submission.csv
```

RF-DETR 1.4.0 uses the COCO category IDs directly here: 1 car, 2 van, 3 truck, 4 bus.
The adapter converts to the common zero-based indices. Category 0 is unused.
The model's per-pass candidate count remains 300. Tiled predictions are merged and
then capped at `--max-det` (default 500). Preserve `run.json` with the weights.
