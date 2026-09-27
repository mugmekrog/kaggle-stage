# %% [markdown]
# # YOLO + RF-DETR — ensemble
# 
# YOLO eğitim notebook'undaki kurulum tamamlanmış olmalı. Yeni notebook kernel'i değişkenleri paylaşmaz: bu hücreleri mevcut YOLO oturumuna kopyalayın veya ENS_PY/ENS_PREPARED yollarını açıkça ayarlayın. Widget seçiminden sonra durup checkpoint ve CSV eşleşmesini kontrol edin; Run all kullanmayın.

# %% [markdown]
# ## setup
# 
# Önceki bölümün ürettiği değişkenleri kullanır.

# %%
from pathlib import Path
from datetime import datetime
import os, sys, json, io, zipfile, subprocess, time
import hashlib
from collections import deque
from google.colab import drive, files
from IPython.display import clear_output, display
import ipywidgets as widgets
drive.mount('/content/drive')
ENS_RESULTS = Path('/content/drive/MyDrive/roketsan_results')
RF_ROOT = Path('/content/drive/MyDrive/roketsan_astra/rfdetr_dinov2')
ENS_DATASET = Path('/content/work/data') if Path('/content/work/data/sample_submission.csv').is_file() else Path('/content/drive/MyDrive/roketsan_dataset')
ENS_PREPARED = Path(globals().get('CUSTOM_PREPARED', globals().get('PREPARED', '/content/vehicles_prepared')))
ENS_PY = globals().get('CUSTOM_PY', globals().get('PY'))
if not ENS_PY or not Path(ENS_PY).is_file():
    candidates = sorted(Path('/content').glob('yolo26l_env_*/bin/python')) + sorted(Path('/content').glob('vehicle_env_yolo26_*/bin/python'))
    assert len(candidates) == 1, f'Mevcut YOLO notebookunda çalıştırın veya ENS_PY yolunu belirtin: {candidates}'
    ENS_PY = str(candidates[0])
ENS_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
ENS_CODE = Path('/content') / ('ensemble_code_' + ENS_STAMP)
ENS_OUTPUT = ENS_RESULTS / ('ensemble_yolo26l_rfdetr_' + ENS_STAMP)
assert (ENS_DATASET / 'sample_submission.csv').is_file()
uploaded = files.upload()  # ensemble_fast_bundle.zip
bundles = []
for name, data in uploaded.items():
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if 'ensemble_vehicle.py' in z.namelist(): bundles.append(data)
assert len(bundles) == 1, 'Tek bir ensemble_fast_bundle.zip seçin.'
ENS_CODE.mkdir()
with zipfile.ZipFile(io.BytesIO(bundles[0])) as z:
    for entry in z.infolist():
        assert (ENS_CODE / entry.filename).resolve().is_relative_to(ENS_CODE.resolve())
    z.extractall(ENS_CODE)
# Sadece küçük CPU fusion bağımlılığı; torch/rfdetr/ultralytics yeniden kurulmaz.
test = subprocess.run([ENS_PY, '-c', 'import ensemble_boxes'], capture_output=True)
if test.returncode:
    subprocess.run([ENS_PY, '-m', 'pip', 'install', 'ensemble-boxes==1.0.9', '--no-deps'], check=True)
    test = subprocess.run([ENS_PY, '-c', 'import ensemble_boxes'], capture_output=True, text=True)
    if test.returncode:
        # ensemble-boxes imports numba; install it only if absent.
        subprocess.run([ENS_PY, '-m', 'pip', 'install', 'numba'], check=True)

# %% [markdown]
# ## choose
# 
# Önceki bölümün ürettiği değişkenleri kullanır.

# %%
def model_run(path):
    for parent in Path(path).parents:
        if (parent / 'best_ap50.json').is_file(): return parent
    return None

def ap50(path):
    root = model_run(path)
    return float(json.loads((root / 'best_ap50.json').read_text())['validation_AP50']) if root else -1

weights = sorted([p for p in ENS_RESULTS.glob('yolo26l_*/best_ap50.pt')
                  if 'smoke' not in p.parent.name and 'audit' not in p.parent.name], key=ap50, reverse=True)
yolo_csvs = sorted([p for p in ENS_RESULTS.glob('yolo26l_*/submission_*/submission.csv')], key=ap50, reverse=True)
rf_csvs = sorted((RF_ROOT / 'submissions').glob('*rfdetr_large_768_t768*final.csv'))
assert rf_csvs, f'RF final CSV bulunamadı: {RF_ROOT / "submissions"}'
assert weights or yolo_csvs, 'YOLO26l checkpoint/CSV bulunamadı; ENS_RESULTS yolunu kontrol edin.'
best_weight = str(weights[0]) if weights else None
best_csv = next((str(p) for p in yolo_csvs if weights and model_run(p) == weights[0].parent), None)
if not weights and yolo_csvs: best_csv = str(yolo_csvs[0])
YOLO_PICK = widgets.Dropdown(options=[('Checkpoint ile YOLO tahmini üret', '__infer__')] +
    [(f'AP50={ap50(p):.4f} | {p}', str(p)) for p in yolo_csvs], value=best_csv or '__infer__',
    description='YOLO CSV', layout=widgets.Layout(width='98%'))
WEIGHT_PICK = widgets.Dropdown(options=[(f'AP50={ap50(p):.4f} | {p}', str(p)) for p in weights],
    description='YOLO .pt', layout=widgets.Layout(width='98%'))
RF_PICK = widgets.Dropdown(options=[(str(p), str(p)) for p in rf_csvs],
    description='RF final', layout=widgets.Layout(width='98%'))
display(YOLO_PICK, WEIGHT_PICK, RF_PICK)
print('Seçimleri kontrol edin. Hazır YOLO CSV varsa GPU inference atlanır.')

# %% [markdown]
# ## runner
# 
# Önceki bölümün ürettiği değişkenleri kullanır.

# %%
def ens_run(*args):
    log_dir = ENS_RESULTS / '_logs' / ENS_OUTPUT.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (str(args[0]) + '_' + datetime.now().strftime('%H%M%S_%f') + '.log')
    command = [str(ENS_PY), '-u', str(ENS_CODE / 'ensemble_vehicle.py'), *map(str, args)]
    tail, last = deque(maxlen=25), time.monotonic()
    print('Log:', log_path, flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        log.write(repr(command) + '\n')
        with subprocess.Popen(command, cwd=ENS_CODE, env=dict(os.environ, MPLBACKEND='Agg'),
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            try:
                for line in process.stdout:
                    log.write(line); log.flush()
                    if line.strip(): tail.append(line)
                    if time.monotonic() - last > 20:
                        clear_output(wait=True); print('Log:', log_path); print(''.join(tail)[-10000:])
                        last = time.monotonic()
                code = process.wait()
            except BaseException:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
    clear_output(wait=True); print('Log:', log_path); print(''.join(tail)[-10000:])
    if code: raise RuntimeError(f'Exit {code}: {log_path}\n' + ''.join(tail)[-10000:])

# %% [markdown]
# ## fusion-now
# 
# Önceki bölümün ürettiği değişkenleri kullanır.

# %%
RF_CSV = Path(RF_PICK.value)
YOLO_WEIGHTS = Path(WEIGHT_PICK.value) if WEIGHT_PICK.value else None
YOLO_IOU, YOLO_CAP = 0.7, 500
if YOLO_WEIGHTS:
    choices = sorted(YOLO_WEIGHTS.parent.glob('nms_validation_*/nms_selection.json'))
    if choices:
        choice = json.loads(choices[-1].read_text())['selected']
        YOLO_IOU, YOLO_CAP = choice['iou'], choice['max_det']
if YOLO_PICK.value == '__infer__':
    assert YOLO_WEIGHTS and YOLO_WEIGHTS.is_file()
    YOLO_PRED_OUTPUT = ENS_RESULTS / ('ensemble_yolo_predictions_' + ENS_STAMP)
    ens_run('predict-yolo', '--weights', YOLO_WEIGHTS, '--dataset', ENS_DATASET,
            '--iou', YOLO_IOU, '--max-det', YOLO_CAP, '--output', YOLO_PRED_OUTPUT)
    YOLO_CSV = YOLO_PRED_OUTPUT / 'yolo.csv'
else:
    YOLO_CSV = Path(YOLO_PICK.value)
    provenance = YOLO_CSV.parent / 'submission_report.json'
    if provenance.is_file():
        saved = json.loads(provenance.read_text())
        if 'selected' in saved and isinstance(saved['selected'], dict):
            YOLO_IOU, YOLO_CAP = saved['selected']['iou'], saved['selected']['max_det']
ens_run('submit', '--yolo', YOLO_CSV, '--rf', RF_CSV, '--dataset', ENS_DATASET, '--output', ENS_OUTPUT)
print('HAZIR ENSEMBLE:', ENS_OUTPUT / 'submission.csv')
print('Bu dosya validation tuning yapılmamış 50/50 ensemble adayıdır. Eski CSVler korunur.')

# %% [markdown]
# ## optional-validation
# 
# Önceki bölümün ürettiği değişkenleri kullanır.

# %%
RUN_SHARED_VALIDATION = False  # Önce hazır submission'ı alın; zaman kalırsa True yapın.
RF_TRAIN_COCO = Path('/content/work/rf_ds/train/_annotations.coco.json')
RF_VAL_CACHE = RF_ROOT / 'preds/rfdetr_large_768_t768__checkpoint_best_total__val.pkl'
RF_FUSION_GRID = RF_ROOT / 'runs/rfdetr_large_768_t768/fusion_grid_val.csv'
if RUN_SHARED_VALIDATION:
    assert YOLO_WEIGHTS and YOLO_WEIGHTS.is_file(), 'Doğru YOLO checkpointini seçin.'
    assert all(p.is_file() for p in (RF_TRAIN_COCO, RF_VAL_CACHE, RF_FUSION_GRID, ENS_PREPARED / 'manifest.json')), 'Split/cache dosyası eksik; hazır 50/50 submission korunuyor.'
    assert YOLO_PICK.value == '__infer__' or model_run(YOLO_CSV) == YOLO_WEIGHTS.parent, 'YOLO validation ve test aynı modelden gelmeli.'
    provenance = YOLO_CSV.parent / ('prediction_provenance.json' if YOLO_PICK.value == '__infer__' else 'submission_report.json')
    assert provenance.is_file(), 'Validation tuning için YOLO CSV checkpoint kaydı gerekli.'
    with YOLO_WEIGHTS.open('rb') as f:
        current_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    assert json.loads(provenance.read_text()).get('checkpoint_sha256') == current_hash, 'YOLO CSV ile seçilen checkpoint eşleşmiyor.'
    ENS_HOLDOUT = ENS_RESULTS / ('ensemble_holdout_' + ENS_STAMP)
    ens_run('holdout', '--data', ENS_PREPARED, '--rf-train-coco', RF_TRAIN_COCO, '--output', ENS_HOLDOUT)
    IDS_FILE = ENS_HOLDOUT / 'common_holdout.json'
    YVAL_OUTPUT = ENS_RESULTS / ('ensemble_yolo_val_' + ENS_STAMP)
    ens_run('predict-yolo', '--weights', YOLO_WEIGHTS, '--dataset', ENS_DATASET, '--ids', IDS_FILE,
            '--iou', YOLO_IOU, '--max-det', YOLO_CAP, '--output', YVAL_OUTPUT)
    RVAL_OUTPUT = ENS_RESULTS / ('ensemble_rf_val_' + ENS_STAMP)
    ens_run('rf-cache', '--ids', IDS_FILE, '--cache', RF_VAL_CACHE, '--fusion-grid', RF_FUSION_GRID, '--output', RVAL_OUTPUT)
    TUNE_OUTPUT = ENS_RESULTS / ('ensemble_tuning_' + ENS_STAMP)
    ens_run('tune', '--data', ENS_PREPARED, '--rf-train-coco', RF_TRAIN_COCO,
            '--yolo', YVAL_OUTPUT / 'yolo.csv', '--rf', RVAL_OUTPUT / 'rf_validation.csv', '--output', TUNE_OUTPUT)
    SELECTED_OUTPUT = ENS_RESULTS / ('ensemble_selected_' + ENS_STAMP)
    ens_run('submit', '--yolo', YOLO_CSV, '--rf', RF_CSV, '--dataset', ENS_DATASET,
            '--selection', TUNE_OUTPUT / 'ensemble_validation.json', '--output', SELECTED_OUTPUT)
    print('RAPOR:', TUNE_OUTPUT / 'ensemble_validation.json')
    print('SEÇİLEN SUBMISSION:', SELECTED_OUTPUT / 'submission.csv')
