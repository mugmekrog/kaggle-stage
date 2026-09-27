# %% [markdown]
# # RF-DETR — mevcut checkpoint ile yeni görüntüler
# 
# Bağımsız yeni Colab oturumunda sırayla çalıştırılabilir. Eğitim/tile veri seti üretilmez. MODEL_DIR ve RUN_NAME eğitimdeki checkpoint klasörünü göstermeli. Sonuç: roketsan_astra/rfdetr_dinov2/boxes.csv. Checkpoint seçimi ve inference cache davranışı ana notebook ile aynıdır.

# %% [markdown]
# ## 1. Ortam
# 
# Eğitim notebook'u ile aynı bağımlılıklar.

# %%
import subprocess
import sys

subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rfdetr[train,augment]', 'ensemble-boxes'], check=True)

# %% [markdown]
# ## 2. Ayarlar
# 
# Önceki eğitimin yollarını ve model ayarlarını kullanın.

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
# ## 3. Drive ve inference importları
# 
# Eğitim akışı atlanır; tile koordinat yardımcıları tanımlanır.

# %%
from google.colab import drive
drive.mount("/content/drive")
import gc, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
for d in (PRED_DIR, SUB_DIR):
    d.mkdir(parents=True, exist_ok=True)

def tile_starts(length, tile, min_overlap):
    if length <= tile:
        return [0]
    ov = int(tile * min_overlap)
    n = math.ceil((length - ov) / (tile - ov))
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]

def make_tiles(w, h, tile=TILE, overlap=OVERLAP):
    return [(x0, y0, min(x0 + tile, w), min(y0 + tile, h))
            for y0 in tile_starts(h, tile, overlap) for x0 in tile_starts(w, tile, overlap)]

# %% [markdown]
# ## 4. Model ve tahmin yardımcıları
# 
# Aynı dört görünüm tahmin yolu.

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
# ## 5. Fusion yardımcıları
# 
# Yeni klasör tahmini için gerekli WBF fonksiyonları.

# %%
# Parallel WBF
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

# %% [markdown]
# ## 6. Yeni görüntüler ve boxes.csv
# 
# NEW_DIR, BOXES_CSV ve BOX_CONF_THR bu bölümün başındadır.

# %%
# 11b) roketsan_dataset/images içindeki yeni görüntüler -> boxes.csv (aynı tile + 4-view TTA + WBF)
NEW_DIR = DRIVE_DATA / "images"
BOXES_CSV = MODEL_DIR / "boxes.csv"
BOX_CONF_THR = 0.001       # 0.0 = mAP için hepsi; temiz/görsel kutu istiyorsan ~0.3 yap

IMG_EXT = {".jpg", ".jpeg", ".png"}
new_paths = {p.stem: p for p in NEW_DIR.iterdir() if p.suffix.lower() in IMG_EXT}
new_ids = sorted(new_paths)
assert new_ids, f"{NEW_DIR} içinde görüntü yok"
print(f"{len(new_ids)} yeni görüntü")

# validation'da en iyi çıkan fusion ayarı varsa onu kullan
grid_csv = OUT_DIR / "fusion_grid_val.csv"
if grid_csv.exists():
    r = pd.read_csv(grid_csv).iloc[0]
    FUSION = {"views": r["views"], "iou": float(r["iou"]), "conf": r["conf"], "full_w": float(r["full_w"])}
else:
    FUSION = DEFAULT_FUSION
print("fusion:", FUSION)

CKPT = best_checkpoint()
m, label_map = load_model(CKPT)
raw_new = predict_split(m, label_map, CKPT, new_paths, new_ids, f"images_{len(new_ids)}")  # Drive'daki preds/ altına cache'lenir
del m; gc.collect(); torch.cuda.empty_cache()

fused = fuse_all([raw_new], new_ids, FUSION, "fusing images")

rows = []
for iid in new_ids:
    b, s, l = fused[iid]
    w, h = raw_new[iid]["wh"]
    for (x1, y1, x2, y2), sc, lb in zip(b, s, l):
        if sc < BOX_CONF_THR:
            continue
        x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
        x2, y2 = min(float(w), float(x2)), min(float(h), float(y2))
        bx, by, bw, bh = round(x1), round(y1), round(x2 - x1), round(y2 - y1)
        if bw * bh < MIN_AREA:
            continue
        rows.append((iid, CLASSES[lb], round(float(sc), 5), bx, by, bw, bh))

boxes = pd.DataFrame(rows, columns=["image_id", "label", "confidence", "x", "y", "w", "h"])
boxes.to_csv(BOXES_CSV, index=False)
print(f"wrote {BOXES_CSV} | {len(boxes)} kutu | {boxes.image_id.nunique()}/{len(new_ids)} görüntüde tespit")
display(boxes.head())
print(boxes["label"].value_counts().to_string())
