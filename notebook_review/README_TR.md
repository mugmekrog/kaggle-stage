# ASTRA notebook inceleme paketi

Başlangıç dosyaları:

| Dosya | İçerik |
|---|---|
| [01_yolo26l_training.ipynb](01_yolo26l_training.ipynb) | Ayarlar, kurulum, veri audit, smoke, yeni eğitim/resume, NMS validation, submission |
| [02_rfdetr_training.ipynb](02_rfdetr_training.ipynb) | Ayarlar, blok split, tile COCO, zaman bütçesi, eğitim, 4-view inference, WBF ve submission |

Her notebook'un aynı isimli `.py` kopyası vardır. Kod review ve satır bazlı diff için bu dosyaları kullanın. `# %%` işaretleri notebook hücrelerini, `# %% [markdown]` işaretleri açıklamaları gösterir. Bunlar bağımsız CLI uygulamaları değildir; Colab/GPU/Drive bağımlılıkları devam eder.

## Kullanım

**YOLO:** A100 Colab oturumunda 01 dosyasını açın. İlk bölümde yolları ve ayarları kontrol edin. İstendiğinde mevcut projenin `cloud/yolo26l_custom_bundle.zip` dosyasını yükleyin. `DO_RESUME=False` yeni eğitim; `DO_RESUME=True` ve açık `RESUME_CHECKPOINT` yolu kesilen eğitime devam içindir. Yeni eğitim ve resume birbirini dışlar. Submission hücresi CSV üretir, yarışmaya yüklemez.

**RF-DETR:** 02 dosyasını ayrı bir A100 Colab oturumunda açın. Ayarlar bölümünde Drive yollarını kontrol edin. Sıralı çalıştırma eğitimden final CSV'ye ilerler. `last.ckpt`/`TRAINING_DONE` ve cache dosyaları önceki run'ın devamı kabul edilir. Yeni deney için run ve yerel veri yollarını ayırın. Bütçe ve paket uyumluluğu hakkında [inceleme notlarını](REVIEW_NOTES.md) okuyun.

Kaynak Downloads dosyaları ve mevcut projenin `cloud/` dosyaları değiştirilmedi. Bu klasör, verilen iki notebook'tan üretilen ayrı bir inceleme kopyasıdır. Çıktılar/loglar, execution count, boş hücreler ve Colab oturum metadata'sı temizlendi. Eski eğitim sonuçları orijinallerde durur.

## Yardımcı akışlar

| Dosya | Önkoşul |
|---|---|
| [03_yolo_rf_ensemble](extras/03_yolo_rf_ensemble.ipynb) | YOLO ortamı + iki modelin checkpoint/CSV dosyaları; widget seçimi sonrası devam |
| [04_ensemble_6040](extras/04_ensemble_6040.ipynb) | 03 ile aynı kernel; metadata/split kurtarma ve ortak validation |
| [05_ensemble_three_models](extras/05_ensemble_three_models.ipynb) | 03 ile aynı kernel + ConvNeXt CSV |
| [06_yolo_new_images](extras/06_yolo_new_images.ipynb) | YOLO kurulumuyla aynı kernel; 40 görüntü; son bölüm için RF boxes.csv |
| [07_rfdetr_new_images](extras/07_rfdetr_new_images.ipynb) | Yeni Colab oturumunda çalışabilir; Drive'da eğitilmiş RF checkpoint'i |

**Ayrı Colab notebook'ları Python değişkenlerini paylaşmaz.** 03–06 dosyalarında aynı kernel gerektiren hücreleri ilgili mevcut oturuma kopyalayın; sadece başka bir notebook sekmesi açmak yeterli değildir. Bu yardımcı akışları ana eğitim notebook'larının Run all yoluna eklemeyin. 03'te widget seçiminden sonra, 06'da isteğe bağlı ensemble öncesinde durun.

`embedded_workers/` kaynak notebook'taki string içine gömülü beş Python dosyasının okunabilir kopyasıdır. Notebook'lar bunların içeriğini kendi içinde taşır; Colab'a ek worker yüklemek gerekmez. Bu klasördeki kopyalar inceleme içindir, notebook'taki kodla otomatik senkronize edilmez.

## İnceleme sırası

1. [Değişiklikler ve mevcut sınırlar](REVIEW_NOTES.md).
2. Ana notebook'ların `.py` kopyaları: ayarlar → veri → eğitim → değerlendirme.
3. YOLO gerçek trainer kodu: projenin `yolo26l_custom/trainer.py`, `run.py`, `infer.py` dosyaları. Notebook yalnızca bundle içindeki bu dosyaları çağırır; kullanılan zip ile yerel kaynak eşleşmesi ayrıca kontrol edilmelidir.
4. Gerekiyorsa `extras/` ve `embedded_workers/`.

`source_manifest.json` kaynak SHA-256 değerlerini, sıfır tabanlı eski hücre numaralarını ve çıktı dosyalarının hash'lerini içerir. Her kod hücresinin `review_source` metadata'sı geldiği eski hücreyi gösterir.

## Doğrulama ve yeniden üretim

`verification.json` statik kontrol sonucudur. Notebook yapısı, her kod hücresinin Python sözdizimi, `.py` eşleşmesi, kaynak hücre kapsamı, korunan kodların AST eşitliği ve yeni eğitim/resume atlama yolları kontrol edilir. GPU eğitimi, paket kurulumu, Drive erişimi, framework API uyumluluğu ve metrik doğruluğu bu kontrolün kapsamı dışındadır.

Python 3.10+ ve standart kütüphane yeterlidir. Proje kökünden:

```powershell
python notebook_review/tools/build_review.py --yolo "C:/.../yolo26l_custom_a100.ipynb" --rf "C:/.../astra_rfdetr_4h (4).ipynb"
python notebook_review/tools/verify_review.py --yolo "C:/.../yolo26l_custom_a100.ipynb" --rf "C:/.../astra_rfdetr_4h (4).ipynb"
```

Builder üretilen notebook ve `.py` dosyalarının üzerine yazar; elle düzenlemeler varsa önce koruyun. Bu ilk temizlik paketinde üretim kaynağı orijinal notebook'lar ve builder'dır. Sonraki kod değişikliklerinde `.ipynb` ve `.py` kopyaları birlikte güncellenmelidir.
