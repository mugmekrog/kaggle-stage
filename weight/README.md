# Eğitilmiş ağırlıklar ve koşu çıktıları

İki modelin kaynak klasör yapısı ayrı dizinlerde korunmuştur:

```text
weight/
  rfdetr_dinov2/
    runs/rfdetr_large_768_t768/
    preds/
    submissions/
    boxes.csv
    ...
  yolo26l_balanced_20260925_221658_750221/
    best_ap50.pt
    last_resume.pt
    nms_validation_20260926_051703_130918/
    submission_20260926_052139_710137/
    ...
  manifest.json
```

## Checkpoint seçimi

| Model | Dosya | Kaynak akıştaki kullanım |
|---|---|---|
| RF-DETR | `rfdetr_dinov2/runs/rfdetr_large_768_t768/checkpoint_best_total.pth` | Notebook'un ilk tercih ettiği inference checkpoint'i |
| RF-DETR | `rfdetr_dinov2/runs/rfdetr_large_768_t768/last.ckpt` | Eğitime devam checkpoint'i |
| YOLO26l | `yolo26l_balanced_20260925_221658_750221/best_ap50.pt` | Kaydedilmiş validation AP50 seçimi |
| YOLO26l | `yolo26l_balanced_20260925_221658_750221/last_resume.pt` | Eğitime devam checkpoint'i |

Her model için yalnızca seçilmiş inference checkpoint'i ve eğitime devam checkpoint'i tutulur. YOLO `weights/` altındaki beş ara epoch ve `best.pt`/`last.pt` dosyaları ile RF-DETR'ın alternatif `checkpoint_best_ema.pth` ve `last_ema.pth` dosyaları güncel daldan çıkarılmıştır. Bu seçim mevcut notebook'ların kullandığı `best_ap50.pt` ve `checkpoint_best_total.pth` yollarını korur; yeni bir model kalitesi karşılaştırması yapılmamıştır.

Tahmin cache'leri, CSV'ler, metrikler, ayarlar ve görseller korunmuştur. `manifest.json` yayımlanan 66 kaynak dosyasının boyutunu ve SHA-256 değerini listeler; çıkarılmış dokuz checkpoint `omitted_checkpoints` alanında kayıtlıdır. Kaynak dosyalarla kopyaların hash'leri karşılaştırılmıştır; modeller bu aktarım sırasında çalıştırılmamıştır.

Bu temizlik güncel checkout/indirme boyutunu azaltır. Önceki commit'ler ve onların LFS nesneleri geçmişte kalır; GitHub LFS depolama kullanımının azaldığı anlamına gelmez.

Python `__pycache__` dosyası hariç tutulmuştur. Kaynak klasörlerin yanındaki ZIP arşivleri aynı içerikleri tekrar etmemesi için eklenmemiştir. Kaynak bilgisayardaki dosyalar değiştirilmemiştir.

## Git LFS ile indirme

Git LFS kurulu bir sistemde:

```bash
git lfs install
git clone https://github.com/mugmekrog/kaggle-stage.git
cd kaggle-stage
git lfs pull
```

Yalnızca bir modelin dosyalarını indirmek için, otomatik LFS indirmesi yapılmadan oluşturulmuş mevcut klonda:

```bash
git lfs pull --include="weight/rfdetr_dinov2/**" --exclude=""
# veya
git lfs pull --include="weight/yolo26l_balanced_20260925_221658_750221/**" --exclude=""
```

`.pt`, `.pth`, `.ckpt`, `.pkl`, CSV, JPG, PNG ve TensorBoard event dosyaları LFS üzerinden tutulur. GitHub kaynak ZIP'inin büyük dosyaları içerdiğini varsaymak yerine Git LFS ile indirin. JSON/YAML/TXT/Python belgeleri normal Git dosyalarıdır.

Model/config dosyaları özgün deneyin ayarlarını ve Drive yollarını taşır. Bunlar otomatik olarak yerel checkout yollarına çevrilmedi; notebook ayarlarında uygun yolları seçin. Model yükleme için özgün paket ortamı gerekir. YOLO custom trainer bundle bağımlılığı hakkında ana README'ye bakın.
