# Değişiklik kaydı ve inceleme notları

## Yapılan düzenlemeler

- İki ana notebook, eğitim ve değerlendirme akışına odaklandı. Eğitim sonrası eklenmiş ensemble/kurtarma/yeni görüntü bölümleri beş yardımcı notebook'a taşındı; hiçbir dolu kaynak kod hücresi kaybolmadı.
- Bölüm başlıkları, girdiler/çıktılar ve oturum önkoşulları eklendi. Uzun hücreler yalnızca üst seviye Python ifadelerinin arasından bölündü; fonksiyon veya kontrol blokları yarılmadı.
- Eski çıktılar, execution count, boş hücreler ve oturum metadata'sı çıkarıldı. Sabit hücre ID'leri, LF satır sonları ve UTF-8 kullanıldı.
- Her notebook için aynı kodu içeren percent-format `.py` eklendi.
- Tek satırdaki uzun gömülü worker string'leri aynı içeriği taşıyan çok satırlı literal olarak görünür hale getirildi; ayrıca `embedded_workers/` içine çıkarıldı. AST eşitliğiyle string içeriğinin korunduğu denetlendi.

## Açıkça yapılan akış değişiklikleri

1. **YOLO resume:** Kaynakta `DO_RESUME=True` ve belirli eski bir koşuya ait mutlak checkpoint yolu vardı. Varsayılan `False`, yol `None` yapıldı. Ayarlar ilk bölüme alındı. Resume seçilirse yol erken doğrulanır ve smoke/yeni eğitim atlanır. Böylece Run all önce yeni eğitim yapıp sonra eski koşuya geçmez. Resume komutu ve eğitim argümanları korunmuştur.
2. **YOLO submission:** `GENERATE_SUBMISSION=True` değeri değiştirilmeden ilk ayarlar bölümüne taşındı.
3. **RF kurulum sırası:** Eğitimden sonra yer alan `rfdetr[train,augment]` kurulumu başlangıca alındı; `rfdetr`/`ensemble-boxes` kurulumu ile birleştirildi. Notebook `!pip` komutları yerine kernel'in `sys.executable -m pip` çağrısı kullanıldı; hata `check=True` ile görünür olur.
4. **RF yeni görüntü akışı:** Kaynaktaki manuel hücre sırası artık ayrı notebook'ta açık bir sıra halinde sunuluyor. Kullanılmayan AP50/submission yardımcıları bu inference notebook'una kopyalanmadı; ana notebook'ta korunuyor.
5. **RF açıklama:** Yerel evaluator'ın pycocotools ile birebir eşit olduğu iddiası kaldırıldı. Hesaplama kodu değiştirilmedi.

Model boyutları, epoch/batch/LR ayarları, loss/trainer çağrıları, seed/split, tile üretimi, NMS/WBF, confidence/alan eşikleri ve CSV biçimleri bu temizlikte yeniden tasarlanmadı. Çıktı temizliği eski skorları yeni doğrulama sonucu gibi sunmaz.

## Kod review sırasında ayrıca değerlendirilmesi gereken mevcut noktalar

Bu maddeler kaynak kod incelemesinden çıkan gözlemlerdir; bu temizlikte algoritmik düzeltme uygulanmadı.

| Yer | Gözlem ve etkisi |
|---|---|
| RF `map50()` + fusion grid | Ground truth olmayan sınıfta `continue` kullanılıyor; dönen AP listesi sonra dört sınıf adıyla `zip` ediliyor. Ara sınıf eksik olduğunda rapor sütunları yanlış sınıf adıyla eşleşebilir. Ortalama da her zaman dört sınıf ortalaması değildir. |
| RF `map50()` | Yerel 101-nokta AP yaklaşımı var; resmi evaluator ile crowd/ignore, max detections, alan aralıkları ve eşit skor sırası davranışı karşılaştırılmadı. |
| RF `predict_split()` | Cache anahtarı checkpoint dosya adı/run/tag kullanıyor; checkpoint içeriği, görüntü içeriği ve inference ayarlarının hash'i yok. Aynı isimde değişen girdiler eski tahminleri döndürebilir. Yeni klasör tag'inde yalnız görüntü sayısı bulunuyor. |
| RF `build_split()` | `_annotations.coco.json` varsa parametre/veri eşleşmesi kontrol edilmeden atlanıyor. Tile/split ayarları değişince aynı DS_DIR kullanımı eski veri setini yeniden kullanabilir. |
| RF `TRAINING_DONE` | Run adı model/çözünürlük/tile'dan türetiliyor. LR, seed veya veri değişse de aynı isim ve bitiş işareti eğitimi atlatabilir. |
| RF kurulum ve TimeBudget | Paket sürümü sabit değil. `rfdetr.training.build_trainer`, Lightning callback, `batch_size="auto"` ve `RFDETR.from_checkpoint` seçilen paket sürümünde birlikte doğrulanmalı. Yerel `rfdetr_dinov2/requirements.txt` içindeki 1.4.0 pin'i bu notebook'a otomatik taşınmadı; aynı API sözleşmesi olduğu varsayılmadı. |
| RF zaman bütçesi | Saat pip kurulumundan sonra başlıyor. Epoch tahmini eğitim süresinden hesaplanıyor; validation/checkpoint ek süresi için ayrı ölçüm yok. Callback'in stop isteği framework davranışına bağlı; kesin 4 saat garantisi değil. |
| RF split/validation | 25'lik leksikografik ID blokları gerçek scene/video bilgisi değildir. Checkpoint validation'ı tile'lar üzerinde ve negatif tile'ların yalnız %5'ini tutarak yapılıyor; fusion seçimi orijinal validation görüntülerini kullanıyor. İki skor aynı değerlendirme protokolü değil. |
| YOLO / RF karşılaştırması | YOLO hazırlığı %20 validation; RF %5 blok validation kullanıyor. Notebook'lardaki skorları doğrudan aynı holdout gibi karşılaştırmayın. Ensemble akışında ortak holdout kontrolünün rolü bu yüzden önemli. |
| RF ID okuma | CSV `image_id` değerleri okunduktan sonra string'e çevriliyor. Baştaki sıfırların anlamlı olduğu yeni veri setlerinde okuma sırasında string dtype verilmesi gerekebilir. |
| RF `boxes.csv` / 40 görüntü fusion | RF dosyasında tespitsiz görüntüler için satır yok. 40 görüntü fusion worker'ı ise tüm görüntü ID'lerinin RF CSV'de olmasını şart koşuyor; tespitsiz görüntü olduğunda akış hata verebilir. |
| RF CSV alan filtresi | Ana submission'da alan filtresi koordinat yuvarlamasından önce uygulanıyor. Sınırdaki kutular için serileştirilmiş alanın ≥200 koşulu ayrıca kontrol edilmeli. |
| YOLO bundle | Gerçek trainer notebook içinde değil, yüklenen zip içinde. Yerel proje dosyalarının review'u kullanılan bundle içeriğinin doğrulandığı anlamına gelmez. |

## Doğrulamanın sınırı

`verification.json` içinde statik kontrollerin sonucu bulunur. Kaynak hash'leri ve kod eşitliği, düzenleme sırasında kayıp/istenmeyen algoritma değişikliği yakalamak içindir. Colab üzerinde kurulum, CUDA forward/backward, tam eğitim, resume checkpoint yükleme ve üretilen CSV'nin yarışma doğrulaması çalıştırılmadı. Bunların başarılı olduğu iddia edilmez.
