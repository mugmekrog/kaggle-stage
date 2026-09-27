# ASTRA — YOLO26l ve RF-DETR notebook'ları

Colab / A100 üzerinde kullanılan eğitim, değerlendirme ve ensemble notebook'larının kod incelemesi için düzenlenmiş kopyaları.

## Ana notebook'lar

| Model | Notebook | Kod inceleme kopyası |
|---|---|---|
| YOLO26l | [Eğitim ve değerlendirme](notebook_review/01_yolo26l_training.ipynb) | [Python](notebook_review/01_yolo26l_training.py) |
| RF-DETR | [Zaman bütçeli eğitim ve değerlendirme](notebook_review/02_rfdetr_training.ipynb) | [Python](notebook_review/02_rfdetr_training.py) |

- [Kullanım ve inceleme rehberi](notebook_review/README_TR.md)
- [Yapılan değişiklikler ve inceleme notları](notebook_review/REVIEW_NOTES.md)
- [Ensemble ve yeni görüntü notebook'ları](notebook_review/extras/)
- [Gömülü worker kaynakları](notebook_review/embedded_workers/)
- [Statik doğrulama sonucu](notebook_review/verification.json)
- [İnceleme paketini ZIP olarak indir](astra_notebook_review.zip)

## Çalıştırma koşulları

Notebook'lar Colab, GPU, veri/checkpoint dosyaları ve ilgili Python paketlerini gerektirir. YOLO eğitim notebook'u ayrıca özgün projenin `yolo26l_custom_bundle.zip` dosyasını ister; bu bağımlılık bu depodaki inceleme ZIP'i değildir ve depoya dahil edilmemiştir. Eğitim veri seti depoda bulunmaz. Eğitilmiş ağırlıklar ve koşu çıktıları [weight/](weight/README.md) altında Git LFS ile sunulur.

Ana notebook'ların ayarlar bölümünden yolları düzenleyin. Yardımcı akışların bazıları aynı Colab kernel'inde oluşturulmuş değişkenleri kullanır; ilgili dosyadaki önkoşulları izleyin.

## Model dosyaları

- [RF-DETR ağırlıkları ve koşu çıktıları](weight/rfdetr_dinov2/)
- [YOLO26l balanced ağırlıkları ve koşu çıktıları](weight/yolo26l_balanced_20260925_221658_750221/)
- [Git LFS ile indirme ve checkpoint rehberi](weight/README.md)

Checkpoint, tahmin, CSV ve görsel dosyalarını almak için Git LFS kurulu bir klonda `git lfs pull` çalıştırın. Notebook'lardaki Drive yollarını kullanacağınız konuma göre ayarlayın.

## İnceleme kapsamı

Kaynak notebook'ların dolu kod hücreleri korunarak akışlar ayrıldı; eski çıktılar ve oturum metadata'sı temizlendi. YOLO yeni eğitim/resume seçimi ve RF kurulum sırası düzenlendi. Ayrıntılar değişiklik notlarında bulunur.

7 notebook, 48 kod hücresi ve `.py` eşleşmeleri statik olarak kontrol edildi. GPU eğitimi ve framework API uyumluluğu bu düzenleme sırasında test edilmedi. Yeniden üretim araçları `notebook_review/tools/` altındadır ve özgün iki notebook'u girdi olarak ister.
