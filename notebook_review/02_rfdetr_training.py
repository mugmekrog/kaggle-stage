# %% [markdown]
# # RF-DETR — zaman bütçeli eğitim ve değerlendirme
# 
# Colab / A100; varsayılan model Large, çözünürlük 768. Kurulum → ayarlar → yerel veri → blok split → tile COCO → eğitim → 4-view tahmin → varsayılan CSV → validation WBF seçimi → final CSV.
# 
# 240 dakikalık bütçe ayarlar hücresinde başlar; pip kurulumu bu saatin dışındadır. 35 dakika inference için ayrılır. Bu süreler garanti değildir. `last.ckpt` varsa resume, `TRAINING_DONE` varsa eğitim atlama davranışı korunmuştur. Yeni deneyde RUN_NAME/WORK yollarını ayırın; mevcut cache ayar değişikliğini denetlemez.

# %% [markdown]
# ## 1. Ortam kurulumu
# 
# Kaynağın iki kurulum hücresi eğitim öncesinde birleştirildi. Sürüm sabitlenmedi; mevcut API uyumluluğu GPU ortamında ayrıca doğrulanmalıdır.

# %%
import subprocess
import sys

subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rfdetr[train,augment]', 'ensemble-boxes'], check=True)

# %% [markdown]
# ## 2. Deney ayarları
# 
# Bütün eğitim, tile ve fusion parametreleri; varsayılan sayısal değerler korundu.

# %%
# 2) Config: edit this cell only
import os, time, math, json, random, pickle, shutil
from pathlib import Path

T0 = time.time()                 # session clock starts here
SESSION_BUDGET_MIN = 240         # total wall time you have
INFER_RESERVE_MIN = 35           # kept free after training for inference, fusion and writing

MYDRIVE = Path("/content/drive/MyDrive")           # "Drive'ım" mounts as MyDrive
DRIVE_DATA = MYDRIVE / "roketsan_dataset"
MODEL_DIR = MYDRIVE / "roketsan_astra" / "rfdetr_dinov2"
WORK = Path("/content/work")

CLASSES = ["car", "van", "truck", "bus"]
SEED = 42
VAL_FRAC = 0.05                  # held-out images for fusion tuning and checkpoint selection
SPLIT_BLOCK = 25                 # consecutive ids per block (video frames stay on one side)

# Tiling (tile == resolution, so pixels are never downscaled)
TILE = 768
OVERLAP = 0.25                   # inference: min overlap, vehicles must fit fully in some tile
TRAIN_OVERLAP = 0.10             # training: fewer tiles, faster epochs
MIN_VIS_FRAC = 0.6               # keep a clipped GT box only if >= 60% of it is inside the tile
NEG_TILE_KEEP = 0.05             # fraction of empty tiles kept
FULL_IMG_LONG_SIDE = 1344        # one downscaled full image per train image (context for big vehicles)
MIN_FULL_BOX_PX = 8              # in full-image copies, drop boxes smaller than 8x8 px at model resolution
RFS_T = 0.25                     # repeat-factor sampling for rare classes (0 disables)

# Model / training
MODEL_SIZE = "large"
RESOLUTION = 768                 # divisible by 32
MAX_EPOCHS = 40                  # upper bound only; the time budget decides when to stop
LR = 1e-4
LOW_LR_EPOCHS = 2                # final epochs at LR x0.1
NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))
AUG = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.15, "contrast_limit": 0.15, "p": 0.4},
}

# Inference / scoring
MIN_AREA = 200
SCORE_THR = 0.001
INFER_BATCH = 16
EDGE_MARGIN = 3
TILE_NMS_IOU = 0.6
VIEW_TOPK = 1000                 # cap per view before fusion (speed)
MAX_DETS = 500
COORD_DECIMALS = 0
N_PROC = max(1, (os.cpu_count() or 2) - 1)
DEFAULT_FUSION = {"views": "all4", "iou": 0.55, "conf": "avg", "full_w": 1.0}

RUN_NAME = f"rfdetr_{MODEL_SIZE}_{RESOLUTION}_t{TILE}"
OUT_DIR = MODEL_DIR / "runs" / RUN_NAME
PRED_DIR = MODEL_DIR / "preds"
SUB_DIR = MODEL_DIR / "submissions"
DS_DIR = WORK / "rf_ds"

def elapsed_min():
    return (time.time() - T0) / 60

random.seed(SEED)
assert RESOLUTION % 32 == 0 and TILE == RESOLUTION

# %% [markdown]
# ## 3. Drive ve yerel veri kopyası
# 
# Görüntüler /content/work altında işlenir; sonuçlar Drive'a yazılır.

# %%
# 3) Mount Drive, copy the dataset to local disk
from google.colab import drive
drive.mount("/content/drive")
for d in (WORK, OUT_DIR, PRED_DIR, SUB_DIR):
    d.mkdir(parents=True, exist_ok=True)

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

assert (DRIVE_DATA / "train" / "annotations.csv").exists(), f"dataset not found under {DRIVE_DATA}"

def copy_tree_parallel(src, dst, workers=32):
    files = [p for p in src.rglob("*") if p.is_file()]
    todo = [(f, dst / f.relative_to(src)) for f in files]
    todo = [(f, t) for f, t in todo if not (t.exists() and t.stat().st_size == f.stat().st_size)]
    for t in {t.parent for _, t in todo}:
        t.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(lambda ft: shutil.copy2(*ft), todo), total=len(todo), unit="file",
                  desc=f"copy to local ({len(files) - len(todo)} already there)"))

DATA_DIR = WORK / "data"
copy_tree_parallel(DRIVE_DATA, DATA_DIR)
print(f"data ready at {DATA_DIR} | elapsed {elapsed_min():.1f} min")

# %% [markdown]
# ## 4. Anotasyonlar ve validation ayrımı
# 
# Sıralı ID'ler 25'lik bloklara ayrılır; validation oranı %5. Bu yöntem scene/video bağımsızlığını kanıtlamaz.

# %%
# 4) Annotations, image index, block split
import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
train_paths = {p.stem: p for p in (DATA_DIR / "train" / "images").iterdir() if p.suffix.lower() in IMG_EXT}
test_paths = {p.stem: p for p in (DATA_DIR / "test" / "images").iterdir() if p.suffix.lower() in IMG_EXT}

def get_sizes(paths):
    def _size(item):
        k, p = item
        with Image.open(p) as im:
            return k, im.size
    with ThreadPoolExecutor(16) as ex:
        return dict(tqdm(ex.map(_size, paths.items()), total=len(paths), desc="image sizes"))

train_sizes, test_sizes = get_sizes(train_paths), get_sizes(test_paths)

ann = pd.read_csv(DATA_DIR / "train" / "annotations.csv")
ann["image_id"] = ann["image_id"].astype(str)
ann["label"] = ann["label"].astype(str).str.strip().str.lower()
n0 = len(ann)
ann = ann[ann["label"].isin(CLASSES) & ann["image_id"].isin(train_paths.keys())].copy()
# clip to image bounds, drop degenerate boxes
W = ann["image_id"].map(lambda i: train_sizes[i][0]); H = ann["image_id"].map(lambda i: train_sizes[i][1])
x1 = ann["x"].clip(0, W); y1 = ann["y"].clip(0, H)
x2 = (ann["x"] + ann["w"]).clip(0, W); y2 = (ann["y"] + ann["h"]).clip(0, H)
ann["x"], ann["y"], ann["w"], ann["h"] = x1, y1, x2 - x1, y2 - y1
ann = ann[(ann["w"] >= 1) & (ann["h"] >= 1)]
ann["cls"] = ann["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
print(f"boxes kept {len(ann)}/{n0}")

GT = {}
for iid, g in ann.groupby("image_id"):
    xy = g[["x", "y"]].to_numpy(np.float32); wh = g[["w", "h"]].to_numpy(np.float32)
    GT[iid] = (np.concatenate([xy, xy + wh], 1), g["cls"].to_numpy(np.int64))
EMPTY_GT = (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
for iid in train_paths:
    GT.setdefault(iid, EMPTY_GT)

print(f"train {len(train_paths)} | test {len(test_paths)} images")
print(ann["label"].value_counts().to_string())

ids_sorted = sorted(train_paths)
blocks = [ids_sorted[i:i + SPLIT_BLOCK] for i in range(0, len(ids_sorted), SPLIT_BLOCK)]
random.Random(SEED).shuffle(blocks)
n_val_blocks = max(1, round(len(blocks) * VAL_FRAC))
val_ids = sorted(i for b in blocks[:n_val_blocks] for i in b)
trn_ids = sorted(i for b in blocks[n_val_blocks:] for i in b)
print(f"split: train {len(trn_ids)} / val {len(val_ids)}")

# %% [markdown]
# ## 5. Tile COCO veri seti
# 
# Tile üretimi, görünürlük filtresi, full-image kopyaları ve train repeat-factor sampling. Mevcut annotation JSON varsa split yeniden üretilmez.

# %%
# 5) Tiled COCO dataset (+ repeat-factor sampling for rare classes)
def tile_starts(length, tile, min_overlap):
    """Fewest evenly spaced tiles covering `length` with at least `min_overlap` overlap."""
    if length <= tile:
        return [0]
    ov = int(tile * min_overlap)
    n = math.ceil((length - ov) / (tile - ov))
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]

def make_tiles(w, h, tile=TILE, overlap=OVERLAP):
    return [(x0, y0, min(x0 + tile, w), min(y0 + tile, h))
            for y0 in tile_starts(h, tile, overlap) for x0 in tile_starts(w, tile, overlap)]

def clip_to_tile(boxes, labels, t, min_vis=MIN_VIS_FRAC):
    x0, y0, x1, y1 = t
    if len(boxes) == 0:
        return boxes, labels
    c = boxes.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(x0, x1); c[:, [1, 3]] = c[:, [1, 3]].clip(y0, y1)
    vis = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (vis / np.maximum(area, 1e-6) >= min_vis) & (c[:, 2] - c[:, 0] >= 2) & (c[:, 3] - c[:, 1] >= 2)
    return c[keep] - np.array([x0, y0, x0, y0], np.float32), labels[keep]

def _tile_one(args):
    iid, split_dir, add_full, neg_keep = args
    r = random.Random(f"{SEED}-{iid}")
    img = Image.open(train_paths[iid]).convert("RGB"); w, h = img.size
    boxes, labels = GT[iid]
    recs = []
    for t in make_tiles(w, h, overlap=TRAIN_OVERLAP):
        b, l = clip_to_tile(boxes, labels, t)
        if len(b) == 0 and r.random() > neg_keep:
            continue
        fn = f"{iid}_{t[0]}_{t[1]}.jpg"
        img.crop(t).save(split_dir / fn, quality=95)
        recs.append((fn, t[2] - t[0], t[3] - t[1], b, l))
    if add_full:
        s = min(1.0, FULL_IMG_LONG_SIDE / max(w, h))
        fw, fh = round(w * s), round(h * s)
        b = boxes * s
        # boxes that become near-invisible once the loader squashes the image to RESOLUTION
        bw = (b[:, 2] - b[:, 0]) * RESOLUTION / fw; bh = (b[:, 3] - b[:, 1]) * RESOLUTION / fh
        k = (bw >= MIN_FULL_BOX_PX) & (bh >= MIN_FULL_BOX_PX)
        fn = f"{iid}_full.jpg"
        (img.resize((fw, fh), Image.BILINEAR) if s < 1 else img).save(split_dir / fn, quality=95)
        recs.append((fn, fw, fh, b[k], labels[k]))
    return recs

def repeat_factors(recs, t):
    n = len(recs)
    if t <= 0 or n == 0:
        return [1.0] * n
    freq = np.zeros(len(CLASSES))
    for *_, l in recs:
        for c in set(l.tolist()):
            freq[c] += 1
    freq = np.maximum(freq / n, 1e-9)
    rc = np.maximum(1.0, np.sqrt(t / freq))
    print("class tile-frequency:", {c: round(float(v), 3) for c, v in zip(CLASSES, freq)},
          "| repeat factor:", {c: round(float(v), 2) for c, v in zip(CLASSES, rc)})
    return [max([1.0] + [rc[c] for c in set(l.tolist())]) for *_, l in recs]

def build_split(ids, name, add_full, neg_keep, rfs_t):
    split_dir = DS_DIR / name
    ann_file = split_dir / "_annotations.coco.json"
    if ann_file.exists():
        print(f"{name}: already built"); return
    split_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    with ThreadPoolExecutor(max(4, os.cpu_count() or 4)) as ex:
        for r in tqdm(ex.map(_tile_one, [(i, split_dir, add_full, neg_keep) for i in ids]),
                      total=len(ids), desc=f"tiling {name}"):
            recs.extend(r)
    rng = random.Random(SEED)
    images, annots = [], []
    for rec, rf in zip(recs, repeat_factors(recs, rfs_t)):
        copies = int(rf) + (1 if rng.random() < rf - int(rf) else 0)
        fn, w, h, b, l = rec
        for _ in range(copies):
            img_id = len(images)
            images.append({"id": img_id, "file_name": fn, "width": int(w), "height": int(h)})
            for bb, ll in zip(b, l):
                bw, bh = float(bb[2] - bb[0]), float(bb[3] - bb[1])
                annots.append({"id": len(annots), "image_id": img_id, "category_id": int(ll) + 1,
                               "bbox": [float(bb[0]), float(bb[1]), bw, bh], "area": bw * bh, "iscrowd": 0})
    cats = [{"id": i + 1, "name": c, "supercategory": c} for i, c in enumerate(CLASSES)]
    ann_file.write_text(json.dumps({"images": images, "annotations": annots, "categories": cats}))
    print(f"{name}: {len(recs)} tiles -> {len(images)} samples, {len(annots)} boxes")

# %%
build_split(trn_ids, "train", add_full=True, neg_keep=NEG_TILE_KEEP, rfs_t=RFS_T)
build_split(val_ids, "valid", add_full=False, neg_keep=NEG_TILE_KEEP, rfs_t=0)
print(f"elapsed {elapsed_min():.1f} min")

# %% [markdown]
# ## 6. Zaman bütçesi ve eğitim
# 
# TimeBudget callback'i, LR düşürme, last.ckpt resume ve bitiş işareti. Callback/framework entegrasyonu gerçek GPU oturumunda kontrol edilmelidir.

# %%
# 6) Train against the clock
import torch
from pytorch_lightning import Callback
import rfdetr.training as rf_training
from rfdetr import RFDETRMedium, RFDETRLarge

class TimeBudget(Callback):
    """Stops at an epoch boundary before `deadline`; drops LR 10x for the last LOW_LR_EPOCHS epochs."""
    def __init__(self, deadline, low_lr_epochs=LOW_LR_EPOCHS, factor=0.1):
        self.deadline, self.low_lr_epochs, self.factor = deadline, low_lr_epochs, factor
        self.durations, self.dropped, self.t_epoch = [], False, None

    def state_dict(self):
        return {"dropped": self.dropped}

    def load_state_dict(self, state):
        self.dropped = bool(state.get("dropped", False))

    def on_train_epoch_start(self, trainer, pl_module):
        self.t_epoch = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if time.time() > self.deadline + 300:  # hard stop, only if an epoch runs far over
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        now = time.time()
        self.durations.append(now - self.t_epoch)
        ep = float(np.mean(self.durations[-2:]))
        fits = int((self.deadline - now) // (ep * 1.03))
        if fits <= 0:
            trainer.should_stop = True
        elif fits <= self.low_lr_epochs and not self.dropped:
            self._drop_lr(trainer)
        tqdm.write(f"[budget] epoch {trainer.current_epoch} took {ep / 60:.1f} min | "
                   f"{(self.deadline - now) / 60:.0f} min left | epochs that still fit: {max(fits, 0)} | "
                   f"low LR: {self.dropped} | stop: {trainer.should_stop}")

    def _drop_lr(self, trainer):
        for cfg in getattr(trainer, "lr_scheduler_configs", []):
            s = cfg.scheduler
            if hasattr(s, "base_lrs"):
                s.base_lrs = [b * self.factor for b in s.base_lrs]
        for opt in trainer.optimizers:
            for g in opt.param_groups:
                g["lr"] *= self.factor
        self.dropped = True
        tqdm.write("[budget] LR dropped x0.1 for the final epochs")

def install_time_budget(deadline):
    orig = getattr(rf_training.build_trainer, "_orig", rf_training.build_trainer)
    def build_trainer(config, model_config, **kw):
        trainer = orig(config, model_config, **kw)
        if kw.get("include_training_callbacks", True):
            trainer.callbacks.append(TimeBudget(deadline))
            print(f"[budget] active, deadline in {(deadline - time.time()) / 60:.0f} min")
        return trainer
    build_trainer._orig = orig
    rf_training.build_trainer = build_trainer

DONE_FLAG = OUT_DIR / "TRAINING_DONE"
if DONE_FLAG.exists():
    print("training already finished for this run, skipping")
else:
    train_min = SESSION_BUDGET_MIN - INFER_RESERVE_MIN - elapsed_min()
    assert train_min > 20, f"only {train_min:.0f} min left for training"
    install_time_budget(time.time() + train_min * 60)
    resume_ckpt = OUT_DIR / "last.ckpt"
    model = {"medium": RFDETRMedium, "large": RFDETRLarge}[MODEL_SIZE]()
    model.train(
        dataset_dir=str(DS_DIR),
        output_dir=str(OUT_DIR),
        resolution=RESOLUTION,
        epochs=MAX_EPOCHS,
        batch_size="auto",
        lr=LR,
        num_workers=NUM_WORKERS,
        aug_config=AUG,
        class_names=CLASSES,
        checkpoint_interval=1000,      # keep only last.ckpt + best .pth on Drive
        early_stopping=False,          # the clock decides
        progress_bar="tqdm",
        tensorboard=True,
        resume=str(resume_ckpt) if resume_ckpt.exists() else None,
    )
    DONE_FLAG.write_text(time.strftime("%F %T"))
    del model

# %%
import gc; gc.collect(); torch.cuda.empty_cache()
print(f"training finished | elapsed {elapsed_min():.1f} min")

# %% [markdown]
# ## 7. Inference yardımcıları
# 
# Checkpoint seçimi, sınıf eşleme, tile NMS, yatay flip ve dört görünüm cache'i.

# %%
# 7) Inference engine: tiles + full image, each with hflip TTA -> 4 raw views per image
from rfdetr import RFDETR
torch.set_float32_matmul_precision("high")

def best_checkpoint():
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "last_ema.pth"):
        if (OUT_DIR / name).exists():
            return OUT_DIR / name
    raise FileNotFoundError(f"no checkpoint in {OUT_DIR}")

def load_model(ckpt):
    m = RFDETR.from_checkpoint(str(ckpt))
    names = [str(n).strip().lower() for n in m.class_names]
    label_map = np.array([CLASSES.index(n) if n in CLASSES else -1 for n in names], np.int64)
    assert sorted(label_map[label_map >= 0].tolist()) == list(range(len(CLASSES))), f"class names: {names}"
    print("loaded", Path(ckpt).name, "| classes:", names)
    return m, label_map

def _empty():
    return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

def nms_np(boxes, scores, labels, iou_thr):
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    b = boxes + labels[:, None].astype(np.float32) * (boxes.max() + 1)
    x1, y1, x2, y2 = b.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter + 1e-9) <= iou_thr]
    return np.array(keep, np.int64)

def topk(v, k=VIEW_TOPK):
    b, s, l = v
    if len(s) <= k:
        return v
    o = np.argpartition(-s, k)[:k]
    return b[o], s[o], l[o]

def run_model(m, label_map, arrays):
    out = []
    for i in range(0, len(arrays), INFER_BATCH):
        dets = m.predict(arrays[i:i + INFER_BATCH], threshold=SCORE_THR)
        for d in (dets if isinstance(dets, list) else [dets]):
            if len(d) == 0:
                out.append(_empty()); continue
            cid = d.class_id.astype(np.int64)
            ok = (cid >= 0) & (cid < len(label_map))
            lab = np.full(len(cid), -1, np.int64); lab[ok] = label_map[cid[ok]]
            k = lab >= 0
            out.append((d.xyxy.astype(np.float32)[k], d.confidence.astype(np.float32)[k], lab[k]))
    return out

def unflip(boxes, width):
    b = boxes.copy(); b[:, 0] = width - boxes[:, 2]; b[:, 2] = width - boxes[:, 0]; return b

def merge_tiles(tile_preds, tiles, w, h):
    B, S, L = [], [], []
    for (b, s, l), (x0, y0, x1, y1) in zip(tile_preds, tiles):
        if len(b) == 0:
            continue
        bad = np.zeros(len(b), bool)   # boxes cut by an internal tile edge; another tile sees them whole
        if x0 > 0: bad |= b[:, 0] <= EDGE_MARGIN
        if y0 > 0: bad |= b[:, 1] <= EDGE_MARGIN
        if x1 < w: bad |= b[:, 2] >= (x1 - x0) - EDGE_MARGIN
        if y1 < h: bad |= b[:, 3] >= (y1 - y0) - EDGE_MARGIN
        B.append(b[~bad] + np.array([x0, y0, x0, y0], np.float32)); S.append(s[~bad]); L.append(l[~bad])
    if not B:
        return _empty()
    b, s, l = np.concatenate(B), np.concatenate(S), np.concatenate(L)
    k = nms_np(b, s, l, TILE_NMS_IOU)
    return topk((b[k], s[k], l[k]))

# %%
def predict_views(m, label_map, img):
    arr = np.asarray(img); h, w = arr.shape[:2]
    tiles = make_tiles(w, h)
    crops = [arr[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles]
    flips = [np.ascontiguousarray(c[:, ::-1]) for c in crops]
    preds = run_model(m, label_map, crops + flips + [arr, np.ascontiguousarray(arr[:, ::-1])])
    n = len(tiles)
    tpf = [(unflip(b, x1 - x0), s, l) for (b, s, l), (x0, y0, x1, y1) in zip(preds[n:2 * n], tiles)]
    fb, fs, fl = preds[-1]
    return {"wh": (w, h), "views": {
        "tile": merge_tiles(preds[:n], tiles, w, h),
        "tile_flip": merge_tiles(tpf, tiles, w, h),
        "full": topk(preds[-2]),
        "full_flip": topk((unflip(fb, w), fs, fl)),
    }}

def predict_split(m, label_map, ckpt, paths, ids, tag):
    out_file = PRED_DIR / f"{RUN_NAME}__{Path(ckpt).stem}__{tag}.pkl"
    if out_file.exists():
        print("cached:", out_file.name); return pickle.loads(out_file.read_bytes())
    raw = {}
    with ThreadPoolExecutor(4) as loader:
        imgs = loader.map(lambda i: Image.open(paths[i]).convert("RGB"), ids)
        for iid, img in tqdm(zip(ids, imgs), total=len(ids), desc=f"predict {tag}"):
            raw[iid] = predict_views(m, label_map, img)
    out_file.write_bytes(pickle.dumps(raw))
    return raw

# %% [markdown]
# ## 8. WBF, yerel AP50 ve CSV
# 
# Fusion ile yerel evaluator ayrı fonksiyonlardır. Yerel AP50'nin resmi evaluator ile birebir eşitliği doğrulanmadı; ayrıntılar REVIEW_NOTES.md içinde.

# %%
# 8) Fusion (parallel WBF) + local mAP@0.5 (local implementation; see REVIEW_NOTES.md)
import multiprocessing as mp
from ensemble_boxes import weighted_boxes_fusion

VIEW_SETS = {"tile+flip": ["tile", "tile_flip"], "all4": ["tile", "tile_flip", "full", "full_flip"]}

def fuse_image(recs, views, iou_thr, conf_type, full_weight):
    w, h = recs[0]["wh"]
    scale = np.array([w, h, w, h], np.float32)
    B, S, L, Wt = [], [], [], []
    for r in recs:
        for v in views:
            b, s, l = r["views"][v]
            B.append((b / scale).clip(0, 1).tolist()); S.append(s.tolist()); L.append(l.tolist())
            Wt.append(full_weight if v.startswith("full") else 1.0)
    if sum(len(x) for x in S) == 0:
        return _empty()
    b, s, l = weighted_boxes_fusion(B, S, L, weights=Wt, iou_thr=iou_thr, skip_box_thr=SCORE_THR, conf_type=conf_type)
    b = (b * scale).astype(np.float32)
    keep = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) >= MIN_AREA
    b, s, l = b[keep], s[keep].astype(np.float32), l[keep].astype(np.int64)
    o = s.argsort()[::-1][:MAX_DETS]
    return b[o], s[o], l[o]

_FCTX = {}
def _fuse_worker(iid):
    c = _FCTX
    return iid, fuse_image([r[iid] for r in c["raws"]], c["views"], c["iou"], c["conf"], c["fw"])

def fuse_all(raw_list, ids, fusion, desc="fusing"):
    _FCTX.update(raws=raw_list, views=VIEW_SETS[fusion["views"]], iou=fusion["iou"],
                 conf=fusion["conf"], fw=fusion["full_w"])
    try:
        with mp.get_context("fork").Pool(N_PROC) as pool:
            return dict(tqdm(pool.imap_unordered(_fuse_worker, ids, chunksize=8), total=len(ids), desc=desc, leave=False))
    except Exception as e:
        print("parallel fusion failed, falling back to serial:", e)
        return dict(_fuse_worker(i) for i in tqdm(ids, desc=desc, leave=False))

def box_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    aa = np.prod(a[:, 2:] - a[:, :2], 1); bb = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)

def map50(preds, gts, n_cls=len(CLASSES), iou_thr=0.5):
    aps = []
    for c in range(n_cls):
        scores, tps, npos = [], [], 0
        for iid, (gb, gl) in gts.items():
            g = gb[gl == c]; npos += len(g)
            pb, ps, pl = preds.get(iid, _empty())
            m = pl == c; p, s = pb[m], ps[m]
            if len(p) == 0:
                continue
            o = s.argsort()[::-1]; p, s = p[o], s[o]
            tp = np.zeros(len(p), np.float32)
            if len(g):
                ious = box_iou(p, g); used = np.zeros(len(g), bool)
                for k in range(len(p)):
                    iou = np.where(used, -1, ious[k]); j = iou.argmax()
                    if iou[j] >= iou_thr:
                        used[j] = True; tp[k] = 1
            scores.append(s); tps.append(tp)
        if npos == 0:
            continue
        if not scores:
            aps.append(0.0); continue
        s = np.concatenate(scores); tp = np.concatenate(tps)
        tp = tp[s.argsort(kind="stable")[::-1]]
        ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
        rec = ctp / npos; prec = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1e-9))[::-1])[::-1]
        idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
        aps.append(float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])))
    return float(np.mean(aps)), aps

# %%
def to_submission(fused, fname):
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub["image_id"] = sub["image_id"].astype(str)
    missing = set(sub["image_id"]) - set(fused)
    assert not missing, f"{len(missing)} test ids without predictions, e.g. {sorted(missing)[:3]}"
    f = f"{{:.{COORD_DECIMALS}f}}"
    rows = []
    for iid in tqdm(sub["image_id"], desc="formatting", leave=False):
        b, s, l = fused[iid]
        w, h = test_sizes[iid]
        parts = []
        for (x1, y1, x2, y2), sc, lb in zip(b, s, l):
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            if (x2 - x1) * (y2 - y1) < MIN_AREA:
                continue
            parts.append(f"{CLASSES[lb]} {min(max(float(sc), 0.0), 1.0):.5f} "
                         f"{f.format(x1)} {f.format(y1)} {f.format(x2 - x1)} {f.format(y2 - y1)}")
        rows.append(" ".join(parts) if parts else "none")
    sub["PredictionString"] = rows
    assert sub["PredictionString"].notna().all() and (sub["PredictionString"].str.len() > 0).all()
    out = SUB_DIR / fname
    sub.to_csv(out, index=False)
    print(f"wrote {out.name} | rows {len(sub)} | 'none' {(sub.PredictionString == 'none').sum()} | "
          f"boxes {sum(len(fused[i][0]) for i in sub.image_id)}")
    return out

# %% [markdown]
# ## 9. Önce test CSV, sonra validation tahmini
# 
# Uzun validation taramasından önce varsayılan fusion ile bir CSV kaydedilir.

# %%
# 9) Predict test FIRST and write a safety submission, then predict validation
test_ids = sorted(test_paths)
CKPT = best_checkpoint()
m, label_map = load_model(CKPT)
raw_test = predict_split(m, label_map, CKPT, test_paths, test_ids, "test")
safety = to_submission(fuse_all([raw_test], test_ids, DEFAULT_FUSION, "fusing test (default)"),
                       f"submission_{RUN_NAME}_default.csv")
print(f"safety submission ready | elapsed {elapsed_min():.1f} min")
raw_val = predict_split(m, label_map, CKPT, train_paths, val_ids, "val")
del m; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 10. Fusion seçimi ve final CSV
# 
# Fusion ayarları validation üzerinde seçilir; sonuçlar fusion_grid_val.csv dosyasına kaydedilir.

# %%
# 10) Tune fusion on validation, write the final submission
gts_val = {i: GT[i] for i in val_ids}
grid = [{"views": v, "iou": iou, "conf": ct, "full_w": fw}
        for v in VIEW_SETS for iou in (0.5, 0.6) for ct in ("avg", "max", "box_and_model_avg")
        for fw in ((1.0,) if v != "all4" else (0.5, 1.0))]
rows = []
for cfg in tqdm(grid, desc="fusion grid"):
    mAP, aps = map50(fuse_all([raw_val], val_ids, cfg), gts_val)
    rows.append({**cfg, "mAP50": mAP, **{f"AP_{c}": a for c, a in zip(CLASSES, aps)}})
res = pd.DataFrame(rows).sort_values("mAP50", ascending=False)
res.to_csv(OUT_DIR / "fusion_grid_val.csv", index=False)
display(res.head(8).round(4))

default_score = map50(fuse_all([raw_val], val_ids, DEFAULT_FUSION), gts_val)[0]
best = res.iloc[0][["views", "iou", "conf", "full_w"]].to_dict()
print(f"val mAP50: default {default_score:.4f} -> best {res.iloc[0]['mAP50']:.4f} with {best}")

final = to_submission(fuse_all([raw_test], test_ids, best, "fusing test (best)"), f"submission_{RUN_NAME}_final.csv")
print(f"DONE | total elapsed {elapsed_min():.1f} min")
# !python "{MYDRIVE}/roketsan_astra/validate_submission.py" "{final}"   # adjust to your checker's CLI

# %% [markdown]
# ## Çıktılar
# 
# `OUT_DIR`: checkpoint'ler, TRAINING_DONE, fusion_grid_val.csv.
# 
# `PRED_DIR`: ham dört görünüm tahminleri. `SUB_DIR`: `_default.csv` ve `_final.csv`.
# 
# Yeni görüntü klasörü için `extras/07_rfdetr_new_images.ipynb` kullanın.
