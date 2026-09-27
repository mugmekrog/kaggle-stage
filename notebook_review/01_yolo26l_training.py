# %% [markdown]
# # YOLO26l — eğitim, doğrulama ve submission
# 
# Colab / A100. Hücreleri sırayla çalıştırın. Ana ayarlar ilk bölümde; eğitim kodu yüklenen `yolo26l_custom_bundle.zip` içindedir. Ensemble ve yeni görüntü akışları `extras/` altında ayrılmıştır.
# 
# **Yeni eğitim:** `DO_RESUME=False`; 1 epoch smoke → 50 epoch eğitim → NMS seçimi → CSV.
# 
# **Devam:** `DO_RESUME=True` ve `RESUME_CHECKPOINT` ayarlayın. Smoke/yeni eğitim atlanır; aynı veriyle checkpoint planına devam edilir. Bitmiş epoch planı uzatılmaz.

# %% [markdown]
# ## 1. Ayarlar ve importlar
# 
# Drive yolları, profil, batch ve eğitim süresini burada düzenleyin.

# %%
from pathlib import Path
from datetime import datetime
import subprocess, sys, os, json, zipfile, io
from google.colab import drive, files
CUSTOM_DATASET = Path('/content/drive/MyDrive/roketsan_dataset')
CUSTOM_RESULTS = Path('/content/drive/MyDrive/roketsan_results')
CUSTOM_PREPARED = Path(globals().get('PREPARED', '/content/vehicles_prepared'))
PROFILE = 'balanced'  # control / balanced / aerial / musgd
BATCH = 8  # A100 başlangıç değeri; OOM durumunda 4 ile yeni smoke deneyin.
IMGSZ = 1280
EPOCHS = 50
REUSE_EXISTING_ENV = True
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
CUSTOM_PROJECT = Path('/content') / ('yolo26l_custom_' + STAMP)
CUSTOM_RUN = CUSTOM_RESULTS / ('yolo26l_' + PROFILE + '_' + STAMP)

# Çalıştırma modu ve isteğe bağlı çıktılar
DO_RESUME = False
RESUME_CHECKPOINT = None  # Devam için Path('/content/drive/MyDrive/.../last_resume.pt')
GENERATE_SUBMISSION = True

# %% [markdown]
# ## 2. Drive ve GPU kontrolü
# 
# Resume yolu yeni eğitim başlamadan kontrol edilir.

# %%
drive.mount('/content/drive')
print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True))
assert (CUSTOM_DATASET / 'train/annotations.csv').is_file(), 'Dataset Drive yolunu düzeltin.'
if DO_RESUME:
    if RESUME_CHECKPOINT is None or not Path(RESUME_CHECKPOINT).is_file():
        raise FileNotFoundError('Geçerli last_resume.pt yolunu ayarlayın.')
    CUSTOM_RUN = Path(RESUME_CHECKPOINT).parent

# %% [markdown]
# ## 3. Eğitim kodunu yükle
# 
# İstendiğinde projenin `cloud/yolo26l_custom_bundle.zip` dosyasını seçin.

# %%
uploaded = files.upload()  # yolo26l_custom_bundle.zip seçin.
bundles = []
for name, payload in uploaded.items():
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if 'yolo26l_custom/trainer.py' in z.namelist() and 'bundle_manifest.json' in z.namelist():
                bundles.append((name, payload))
assert len(bundles) == 1, 'Tek bir yolo26l_custom_bundle.zip yükleyin.'
assert not CUSTOM_PROJECT.exists(), 'Kurulum hücresini yeniden çalıştırıp yeni klasör oluşturun.'
CUSTOM_PROJECT.mkdir()
with zipfile.ZipFile(io.BytesIO(bundles[0][1])) as z:
    for member in z.infolist():
        dest = (CUSTOM_PROJECT / member.filename).resolve()
        assert dest.is_relative_to(CUSTOM_PROJECT.resolve()), 'Geçersiz zip yolu.'
    z.extractall(CUSTOM_PROJECT)
print('Kod:', CUSTOM_PROJECT)

# %% [markdown]
# ## 4. Python ortamı
# 
# Mevcut PY ortamı varsa kullanılır; aksi halde izole Python 3.11 ortamı oluşturulur.

# %%
old_py = globals().get('PY')
if REUSE_EXISTING_ENV and old_py and Path(old_py).is_file():
    CUSTOM_PY = str(old_py)
    print('Mevcut ortam kullanılıyor:', CUSTOM_PY)
else:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run([sys.executable, '-m', 'uv', 'python', 'install', '3.11'], check=True)
    env = Path('/content') / ('yolo26l_env_' + STAMP)
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', str(env)], check=True)
    CUSTOM_PY = str(env / 'bin/python')
    subprocess.run([CUSTOM_PY, '-m', 'pip', 'install', 'torch==2.5.1', 'torchvision==0.20.1',
                    '--index-url', 'https://download.pytorch.org/whl/cu121'], check=True)
    subprocess.run([CUSTOM_PY, '-m', 'pip', 'install', '-r',
                    str(CUSTOM_PROJECT / 'yolo26l_custom/requirements.txt')], check=True)
subprocess.run([CUSTOM_PY, '-c',
    'from yolo26l_custom.run import library_check; import torch; '
    'print(library_check()); assert torch.cuda.is_available(), "GPU gerekli"'],
    cwd=CUSTOM_PROJECT, env=dict(os.environ, MPLBACKEND='Agg'), check=True)

# %% [markdown]
# ## 5. Log ve komut çalıştırıcısı
# 
# Tam log Drive'a, sınırlı son satırlar hücre çıktısına yazılır.

# %%
from collections import deque
import time
from IPython.display import clear_output

def custom_run(script, *args):
    log_dir = CUSTOM_RESULTS / '_logs' / CUSTOM_RUN.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (Path(script).stem + '_' + datetime.now().strftime('%H%M%S_%f') + '.log')
    command = [CUSTOM_PY, '-u', str(CUSTOM_PROJECT / script), *map(str, args)]
    tail, last = deque(maxlen=30), time.monotonic()
    print('Tam log:', log_path, flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        log.write(repr(command) + '\n')
        log.flush()
        with subprocess.Popen(command, cwd=CUSTOM_PROJECT,
             env=dict(os.environ, MPLBACKEND='Agg', YOLO_VERBOSE='False'),
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
             encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.strip(): tail.append(line)
                    if time.monotonic() - last >= 30:
                        clear_output(wait=True)
                        print('Tam log:', log_path)
                        print(''.join(tail)[-10000:])
                        last = time.monotonic()
                code = process.wait()
            except BaseException:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    clear_output(wait=True)
    print('Tam log:', log_path)
    print(''.join(tail)[-10000:])
    if code:
        raise RuntimeError(f'Exit {code}. Log: {log_path}\n' + ''.join(tail)[-10000:])

# %% [markdown]
# ## 6. Veri hazırlığı ve audit
# 
# Var olan manifest korunur. Yeni hazırlık: seed=42, validation=%20.

# %%
if not (CUSTOM_PREPARED / 'manifest.json').is_file():
    custom_run('prepare_data.py', '--dataset', CUSTOM_DATASET, '--output', CUSTOM_PREPARED,
               '--seed', '42', '--val-fraction', '0.2')
custom_run('check_data.py', '--data', CUSTOM_PREPARED)
AUDIT_RUN = CUSTOM_RESULTS / ('yolo26l_audit_' + STAMP)
custom_run('yolo26l_custom/run.py', 'train', '--data', CUSTOM_PREPARED, '--output', AUDIT_RUN,
           '--profile', PROFILE, '--imgsz', IMGSZ, '--batch', BATCH, '--epochs', EPOCHS, '--dry-run')
audit = json.loads((AUDIT_RUN / 'data_audit.json').read_text())
for split in ('train', 'valid'):
    print(split, audit[split]['class_counts'], 'max instances:', audit[split]['max_instances_per_image'])
print('Scene-grouped:', audit['scene_grouped'])

# %% [markdown]
# ## 7. Smoke testi
# 
# Yalnızca yeni eğitim modunda. Başarılı checkpoint üretimi kontrol edilir.

# %%
if not DO_RESUME:
    SMOKE_OK = False
    SMOKE_RUN = CUSTOM_RESULTS / ('yolo26l_smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    custom_run('yolo26l_custom/run.py', 'train', '--data', CUSTOM_PREPARED, '--output', SMOKE_RUN,
               '--profile', PROFILE, '--epochs', '1', '--imgsz', IMGSZ, '--batch', BATCH, '--workers', '2')
    assert (SMOKE_RUN / 'best_ap50.pt').is_file() and (SMOKE_RUN / 'last_resume.pt').is_file()
    print('Uygulanan ağırlıklar:', (SMOKE_RUN / 'class_weights.json').read_text())
    SMOKE_OK = True
    SMOKE_SETTINGS = (PROFILE, IMGSZ, BATCH)

# %% [markdown]
# ## 8. Tam eğitim
# 
# Yalnızca yeni eğitim modunda. Smoke ağırlıklarından devam edilmez.

# %%
if not DO_RESUME:
    assert SMOKE_OK and SMOKE_SETTINGS == (PROFILE, IMGSZ, BATCH), 'Bu ayarlarla önce smoke çalıştırın.'
    # Bu hücre fresh pretrained YOLO26l'den başlar; smoke checkpoint'inden devam etmez.
    if CUSTOM_RUN.exists():
        CUSTOM_RUN = CUSTOM_RESULTS / ('yolo26l_' + PROFILE + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    custom_run('yolo26l_custom/run.py', 'train', '--data', CUSTOM_PREPARED, '--output', CUSTOM_RUN,
               '--profile', PROFILE, '--epochs', EPOCHS, '--imgsz', IMGSZ, '--batch', BATCH, '--workers', '2')
    print('En iyi:', CUSTOM_RUN / 'best_ap50.pt')

# %% [markdown]
# ## 9. Checkpoint'ten devam
# 
# Yalnızca DO_RESUME=True olduğunda çalışır.

# %%
if DO_RESUME:
    CUSTOM_RUN = Path(RESUME_CHECKPOINT).parent
    custom_run('yolo26l_custom/run.py', 'resume', '--checkpoint', RESUME_CHECKPOINT,
               '--data', CUSTOM_PREPARED, '--workers', '2')

# %% [markdown]
# ## 10. Validation ve NMS seçimi
# 
# Eğitim veya resume tamamlandıktan sonra seçilen run değerlendirilir.

# %%
# Eğitim tamamen bittikten/durduktan sonra çalıştırın.
CUSTOM_WEIGHTS = CUSTOM_RUN / 'best_ap50.pt'
assert CUSTOM_WEIGHTS.is_file()
CUSTOM_VALIDATION = CUSTOM_RUN / ('nms_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
custom_run('yolo26l_custom/infer.py', 'validate', '--weights', CUSTOM_WEIGHTS,
           '--data', CUSTOM_PREPARED, '--output', CUSTOM_VALIDATION)
print('Paylaşılacak rapor:', CUSTOM_VALIDATION / 'nms_selection.json')

# %% [markdown]
# ## 11. Submission CSV
# 
# GENERATE_SUBMISSION ile kontrol edilir. Dosya üretilir; otomatik yükleme yapılmaz.

# %%
if GENERATE_SUBMISSION:
    CUSTOM_SUBMISSION = CUSTOM_RUN / ('submission_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    custom_run('yolo26l_custom/infer.py', 'submit', '--weights', CUSTOM_WEIGHTS,
               '--data', CUSTOM_PREPARED, '--dataset', CUSTOM_DATASET,
               '--selection', CUSTOM_VALIDATION / 'nms_selection.json', '--output', CUSTOM_SUBMISSION)
    print('Yarışmaya gönder:', CUSTOM_SUBMISSION / 'submission.csv')
