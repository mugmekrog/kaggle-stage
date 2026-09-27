"""Build review notebooks without running training or importing ML frameworks."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = {"sources": {}, "notebooks": {}, "embedded_workers": {}}


def source(cell):
    return "".join(cell["source"])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    nb = json.loads(path.read_text(encoding="utf-8"))
    MANIFEST["sources"][name] = {
        "filename": path.name,
        "sha256": digest(path),
        "cells": len(nb["cells"]),
        "nonempty_code_cells": [
            i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and source(c).strip()
        ],
    }
    return nb["cells"]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def readable_workers(text, origin):
    """Render embedded worker strings as readable multiline literals; verify AST equality."""
    tree = ast.parse(text)
    lines = text.splitlines(True)
    replacements = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and len(node.args[0].value) > 1000):
            continue
        literal = node.args[0]
        value = literal.value
        ast.parse(value)
        name = f"{origin}_{node.func.value.id.lower()}.py"
        worker = ROOT / "embedded_workers" / name
        worker.parent.mkdir(parents=True, exist_ok=True)
        worker.write_text(value, encoding="utf-8", newline="\n")
        MANIFEST["embedded_workers"][worker.relative_to(ROOT).as_posix()] = digest(worker)
        # Use a raw triple-quoted literal only if it encodes the exact original string.
        candidate = "r'''" + value + "'''"
        if "'''" in value or ast.literal_eval(candidate) != value:
            raise ValueError(f"Cannot safely render embedded worker: {name}")
        start = sum(map(len, lines[:literal.lineno - 1])) + len(
            lines[literal.lineno - 1].encode()[:literal.col_offset].decode())
        end = sum(map(len, lines[:literal.end_lineno - 1])) + len(
            lines[literal.end_lineno - 1].encode()[:literal.end_col_offset].decode())
        replacements.append((start, end, candidate))
    for start, end, value in sorted(replacements, reverse=True):
        text = text[:start] + value + text[end:]
    assert ast.dump(tree) == ast.dump(ast.parse(text))
    return text


def chunks(text, limit=65):
    """Split only between top-level statements; never break suites or literal strings."""
    tree = ast.parse(text)
    lines = text.splitlines(True)
    boundaries = [0]
    start = 0
    for node in tree.body:
        node_start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        if node_start - start >= limit:
            # Keep immediately preceding comments with the following statement.
            while node_start > start and (not lines[node_start - 1].strip()
                                         or lines[node_start - 1].lstrip().startswith("#")):
                node_start -= 1
            if node_start > start:
                boundaries.append(node_start)
                start = node_start
    boundaries.append(len(lines))
    parts = ["".join(lines[a:b]).strip() + "\n" for a, b in zip(boundaries, boundaries[1:])]
    assert ast.dump(tree) == ast.dump(ast.parse("\n".join(parts)))
    return parts


class Notebook:
    def __init__(self, name, title, intro):
        self.name = name
        self.cells = [markdown(f"# {title}\n\n{intro}")]
        self.origins = set()

    def section(self, title, note, text, origin):
        self.cells.append(markdown(f"## {title}\n\n{note}"))
        if not text.strip():
            return
        self.origins.add(origin)
        text = readable_workers(text, origin.replace(":", "_"))
        for part in chunks(text):
            self.cells.append({
                "cell_type": "code", "execution_count": None, "outputs": [],
                "metadata": {"review_source": origin}, "source": part.splitlines(True),
            })

    def save(self):
        for i, cell in enumerate(self.cells):
            cell["id"] = f"cell-{i:03d}"
        nb = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
                "accelerator": "GPU", "colab": {"provenance": []},
            }, "nbformat": 4, "nbformat_minor": 5,
        }
        path = ROOT / (self.name + ".ipynb")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        script = []
        for c in self.cells:
            if c["cell_type"] == "markdown":
                script.append("# %% [markdown]\n" + "\n".join("# " + s for s in source(c).splitlines()))
            else:
                script.append("# %%\n" + source(c).rstrip())
        py = ROOT / (self.name + ".py")
        py.write_text("\n\n".join(script) + "\n", encoding="utf-8", newline="\n")
        compile(py.read_text(encoding="utf-8"), str(py), "exec")
        MANIFEST["notebooks"][self.name] = {
            "source_cells": sorted(self.origins), "cells": len(self.cells),
            "notebook_sha256": digest(path), "python_sha256": digest(py),
        }


def yolo_main(cells):
    nb = Notebook("01_yolo26l_training", "YOLO26l — eğitim, doğrulama ve submission",
        "Colab / A100. Hücreleri sırayla çalıştırın. Ana ayarlar ilk bölümde; "
        "eğitim kodu yüklenen `yolo26l_custom_bundle.zip` içindedir. "
        "Ensemble ve yeni görüntü akışları `extras/` altında ayrılmıştır.\n\n"
        "**Yeni eğitim:** `DO_RESUME=False`; 1 epoch smoke → 50 epoch eğitim → NMS seçimi → CSV.\n\n"
        "**Devam:** `DO_RESUME=True` ve `RESUME_CHECKPOINT` ayarlayın. Smoke/yeni eğitim atlanır; "
        "aynı veriyle checkpoint planına devam edilir. Bitmiş epoch planı uzatılmaz.")
    config = source(cells[1]).replace("drive.mount('/content/drive')\n", "")
    config = config[:config.index("print(subprocess.check_output")]
    config += "\n# Çalıştırma modu ve isteğe bağlı çıktılar\nDO_RESUME = False\n"
    config += "RESUME_CHECKPOINT = None  # Devam için Path('/content/drive/MyDrive/.../last_resume.pt')\n"
    config += "GENERATE_SUBMISSION = True\n"
    nb.section("1. Ayarlar ve importlar", "Drive yolları, profil, batch ve eğitim süresini burada düzenleyin.", config, "yolo:1")
    nb.section("2. Drive ve GPU kontrolü", "Resume yolu yeni eğitim başlamadan kontrol edilir.",
        "drive.mount('/content/drive')\n" + source(cells[1])[source(cells[1]).index("print(subprocess.check_output"):]
        + "\nif DO_RESUME:\n    if RESUME_CHECKPOINT is None or not Path(RESUME_CHECKPOINT).is_file():\n"
        + "        raise FileNotFoundError('Geçerli last_resume.pt yolunu ayarlayın.')\n"
        + "    CUSTOM_RUN = Path(RESUME_CHECKPOINT).parent\n", "yolo:1")
    sections = {
        2: ("3. Eğitim kodunu yükle", "İstendiğinde projenin `cloud/yolo26l_custom_bundle.zip` dosyasını seçin."),
        3: ("4. Python ortamı", "Mevcut PY ortamı varsa kullanılır; aksi halde izole Python 3.11 ortamı oluşturulur."),
        4: ("5. Log ve komut çalıştırıcısı", "Tam log Drive'a, sınırlı son satırlar hücre çıktısına yazılır."),
        5: ("6. Veri hazırlığı ve audit", "Var olan manifest korunur. Yeni hazırlık: seed=42, validation=%20."),
        6: ("7. Smoke testi", "Yalnızca yeni eğitim modunda. Başarılı checkpoint üretimi kontrol edilir."),
        7: ("8. Tam eğitim", "Yalnızca yeni eğitim modunda. Smoke ağırlıklarından devam edilmez."),
        9: ("9. Checkpoint'ten devam", "Yalnızca DO_RESUME=True olduğunda çalışır."),
        10: ("10. Validation ve NMS seçimi", "Eğitim veya resume tamamlandıktan sonra seçilen run değerlendirilir."),
        11: ("11. Submission CSV", "GENERATE_SUBMISSION ile kontrol edilir. Dosya üretilir; otomatik yükleme yapılmaz."),
    }
    for i, (title, note) in sections.items():
        text = source(cells[i])
        if i in (6, 7):
            text = "if not DO_RESUME:\n" + "\n".join("    " + line if line else "" for line in text.splitlines())
        elif i == 9:
            text = text[text.index("if DO_RESUME:"):]
        elif i == 11:
            text = text.replace("GENERATE_SUBMISSION = True\n", "")
        nb.section(title, note, text, f"yolo:{i}")
    nb.save()


def rf_main(cells):
    nb = Notebook("02_rfdetr_training", "RF-DETR — zaman bütçeli eğitim ve değerlendirme",
        "Colab / A100; varsayılan model Large, çözünürlük 768. "
        "Kurulum → ayarlar → yerel veri → blok split → tile COCO → eğitim → "
        "4-view tahmin → varsayılan CSV → validation WBF seçimi → final CSV.\n\n"
        "240 dakikalık bütçe ayarlar hücresinde başlar; pip kurulumu bu saatin dışındadır. "
        "35 dakika inference için ayrılır. Bu süreler garanti değildir. "
        "`last.ckpt` varsa resume, `TRAINING_DONE` varsa eğitim atlama davranışı korunmuştur. "
        "Yeni deneyde RUN_NAME/WORK yollarını ayırın; mevcut cache ayar değişikliğini denetlemez.")
    nb.section("1. Ortam kurulumu", "Kaynağın iki kurulum hücresi eğitim öncesinde birleştirildi. "
        "Sürüm sabitlenmedi; mevcut API uyumluluğu GPU ortamında ayrıca doğrulanmalıdır.",
        "import subprocess\nimport sys\n\n"
        "subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=True)\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rfdetr[train,augment]', 'ensemble-boxes'], check=True)\n",
        "rf:1+8")
    descriptions = {
        2: ("2. Deney ayarları", "Bütün eğitim, tile ve fusion parametreleri; varsayılan sayısal değerler korundu."),
        3: ("3. Drive ve yerel veri kopyası", "Görüntüler /content/work altında işlenir; sonuçlar Drive'a yazılır."),
        4: ("4. Anotasyonlar ve validation ayrımı", "Sıralı ID'ler 25'lik bloklara ayrılır; validation oranı %5. Bu yöntem scene/video bağımsızlığını kanıtlamaz."),
        5: ("5. Tile COCO veri seti", "Tile üretimi, görünürlük filtresi, full-image kopyaları ve train repeat-factor sampling. "
            "Mevcut annotation JSON varsa split yeniden üretilmez."),
        7: ("6. Zaman bütçesi ve eğitim", "TimeBudget callback'i, LR düşürme, last.ckpt resume ve bitiş işareti. "
            "Callback/framework entegrasyonu gerçek GPU oturumunda kontrol edilmelidir."),
        9: ("7. Inference yardımcıları", "Checkpoint seçimi, sınıf eşleme, tile NMS, yatay flip ve dört görünüm cache'i."),
        10: ("8. WBF, yerel AP50 ve CSV", "Fusion ile yerel evaluator ayrı fonksiyonlardır. "
             "Yerel AP50'nin resmi evaluator ile birebir eşitliği doğrulanmadı; ayrıntılar REVIEW_NOTES.md içinde."),
        11: ("9. Önce test CSV, sonra validation tahmini", "Uzun validation taramasından önce varsayılan fusion ile bir CSV kaydedilir."),
        12: ("10. Fusion seçimi ve final CSV", "Fusion ayarları validation üzerinde seçilir; sonuçlar fusion_grid_val.csv dosyasına kaydedilir."),
    }
    for i, (title, note) in descriptions.items():
        text = source(cells[i])
        if i == 10:
            text = text.replace("(per-class AP, mean of 4; matches pycocotools at IoU 0.5)", "(local implementation; see REVIEW_NOTES.md)")
        nb.section(title, note, text, f"rf:{i}")
    nb.cells.append(markdown("## Çıktılar\n\n`OUT_DIR`: checkpoint'ler, TRAINING_DONE, fusion_grid_val.csv.\n\n"
        "`PRED_DIR`: ham dört görünüm tahminleri. `SUB_DIR`: `_default.csv` ve `_final.csv`.\n\n"
        "Yeni görüntü klasörü için `extras/07_rfdetr_new_images.ipynb` kullanın."))
    nb.save()


def extras(yolo, rf):
    specs = [
        ("03_yolo_rf_ensemble", "YOLO + RF-DETR — ensemble", [13],
         "YOLO eğitim notebook'undaki kurulum tamamlanmış olmalı. Yeni notebook kernel'i değişkenleri paylaşmaz: "
         "bu hücreleri mevcut YOLO oturumuna kopyalayın veya ENS_PY/ENS_PREPARED yollarını açıkça ayarlayın. "
         "Widget seçiminden sonra durup checkpoint ve CSV eşleşmesini kontrol edin; Run all kullanmayın."),
        ("04_ensemble_6040", "Ensemble — metadata kurtarma ve 60/40 karşılaştırması", [19],
         "03 ensemble akışının devamıdır. Hücreleri aynı Colab kernel'ine kopyalayın. "
         "ENS_*, YOLO_WEIGHTS, YOLO_CSV, RF_CSV ve ens_run tanımlı olmalı. "
         "Ortak validation verisiyle iki aday karşılaştırılır."),
        ("05_ensemble_three_models", "ConvNeXt + YOLO + RF-DETR — sabit ağırlıklı ensemble", [23],
         "03 ensemble akışının devamıdır; aynı kernel gereklidir. ConvNeXt CSV yolunu kontrol edin. "
         "Ağırlıklar %50 / %25 / %25; validation tuning yapılmaz."),
        ("06_yolo_new_images", "YOLO — yeni görüntüler ve RF ile 50/50 ensemble", [25, 28, 30],
         "YOLO kurulum hücreleri aynı kernel'de çalışmış olmalı; bu hücreleri o oturuma kopyalayın. "
         "Varsayılan klasör roketsan_dataset/images, beklenen görüntü sayısı 40. "
         "İlk iki bölüm yalnız YOLO tahmini üretir; son bölüm RF boxes.csv gerektiren isteğe bağlı ensemble'dır. "
         "Run all kullanmayın; son bölümü yalnız iki model çıktısı hazırsa çalıştırın."),
    ]
    labels = {13: "Kurulum, model seçimi ve fusion", 19: "Metadata, split kurtarma ve karşılaştırma",
              23: "Üç model fusion", 25: "Checkpoint seçimi", 28: "Yeni görüntü tahmini", 30: "İsteğe bağlı RF + YOLO fusion"}
    for name, title, indices, intro in specs:
        nb = Notebook("extras/" + name, title, intro)
        for i in indices:
            text = source(yolo[i])
            # Existing percent-cell markers preserve natural widget/runner boundaries.
            parts = re.split(r"(?m)^# %% ([^\n]+)\n", text)
            if len(parts) > 1:
                for label, body in zip(parts[1::2], parts[2::2]):
                    nb.section(label, "Önceki bölümün ürettiği değişkenleri kullanır.", body, f"yolo:{i}")
            else:
                nb.section(labels[i], "Gömülü worker kaynakları ayrıca embedded_workers/ altında okunabilir.", text, f"yolo:{i}")
        nb.save()
    nb = Notebook("extras/07_rfdetr_new_images", "RF-DETR — mevcut checkpoint ile yeni görüntüler",
        "Bağımsız yeni Colab oturumunda sırayla çalıştırılabilir. Eğitim/tile veri seti üretilmez. "
        "MODEL_DIR ve RUN_NAME eğitimdeki checkpoint klasörünü göstermeli. "
        "Sonuç: roketsan_astra/rfdetr_dinov2/boxes.csv. "
        "Checkpoint seçimi ve inference cache davranışı ana notebook ile aynıdır.")
    main = json.loads((ROOT / "02_rfdetr_training.ipynb").read_text(encoding="utf-8"))
    setup = next(source(c) for c in main["cells"] if c["cell_type"] == "code")
    nb.section("1. Ortam", "Eğitim notebook'u ile aynı bağımlılıklar.", setup, "rf:1+8")
    nb.section("2. Ayarlar", "Önceki eğitimin yollarını ve model ayarlarını kullanın.", source(rf[2]), "rf:2")
    restore = "\n".join(source(rf[14]).splitlines()[2:])
    nb.section("3. Drive ve inference importları", "Eğitim akışı atlanır; tile koordinat yardımcıları tanımlanır.", restore, "rf:14")
    nb.section("4. Model ve tahmin yardımcıları", "Aynı dört görünüm tahmin yolu.", source(rf[9]), "rf:9")
    # Submission/scoring helpers are unused for folder prediction, so omit them here.
    fusion = source(rf[10]).split("def box_iou(")[0]
    fusion = fusion.replace("# 8) Fusion (parallel WBF) + local mAP@0.5 (per-class AP, mean of 4; matches pycocotools at IoU 0.5)", "# Parallel WBF")
    nb.section("5. Fusion yardımcıları", "Yeni klasör tahmini için gerekli WBF fonksiyonları.", fusion, "rf:10")
    nb.section("6. Yeni görüntüler ve boxes.csv", "NEW_DIR, BOXES_CSV ve BOX_CONF_THR bu bölümün başındadır.", source(rf[15]), "rf:15")
    nb.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    args = parser.parse_args()
    yolo = load(args.yolo, "yolo")
    rf = load(args.rf, "rf")
    yolo_main(yolo)
    rf_main(rf)
    extras(yolo, rf)
    (ROOT / "source_manifest.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Built {len(MANIFEST['notebooks'])} notebooks and paired Python exports.")


if __name__ == "__main__":
    main()
