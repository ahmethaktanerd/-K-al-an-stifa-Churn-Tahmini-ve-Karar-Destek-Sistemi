---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #F8FAFC
color: #1E293B
---

# 💼 İK Çalışan İstifa (Churn) Tahmini
## 📊 Karar Destek Sistemi Sunumu

**Proje Seçeneği:** Classification (Sınıflandırma)
**Kısa Problem:** Kritik personeller şirketi terk etmeden önce tespit edilerek yıllık milyon dolarlık işe alım ve onboarding maliyetlerinden tasarruf sağlanması.
**Ekip Üyeleri:** () ve Veri Bilimi Ekibi

---

# 📖 1. Veri Hikayesi (Data Story)

- **Veri Kaynağı:** Şirket İK veritabanından alınan boylamsal panel veri yapısı (49,653 satır). Her çalışanın yıllara göre demografik ve operasyonel bilgilerini içerir.
- **Gerçek İhtiyaç:** Hangi çalışanın istifa etme olasılığının yüksek olduğunu önceden kestirememek, İK'nın reaktif ve verimsiz prim/zam dağıtmasına yol açmaktadır.
- **Neden Önemli?:** Organizasyonel bilgi birikimini şirkette tutmak, departman verimliliklerini korumak ve çalışan başına **$15,000** olan ikame maliyetini engellemek için veri odaklı karar alabilmek son derece kritiktir.

---

# 👥 2. Ekip Rolleri & Sorumluluklar

- **HR Director (Proje Sponsoru):** İş probleminin tanımlanması, İK elde tutma bütçesinin ($3,000/kişi) yönetimi ve elde tutma aksiyonlarının koordinasyonu.
- **Veri Bilimci ():** Veri temizleme, EDA (keşifsel analiz), grup tabanlı split tasarımı ve veri sızıntısının engellenmesi sorumluluğu.
- **ML Mühendisi:** 10 farklı modelin eğitimi, çapraz doğrulama (CV), hiperparametre optimizasyonu ve modellerin `.joblib` formatında paketlenmesi.
- **Arayüz & Raporlama Geliştirici:** Streamlit tabanlı çalışan karar destek arayüzünün yazılması ve PDF final raporlama altyapısının kurulması.

---

# ⚖️ 3. Karar Değeri & Sınırlılıklar

- **Desteklenen Karar:** Modelin istifa riski skorlaması sayesinde İK, bütçeyi kime harcayacağına karar verir. İstifa riski **>%80** olan çalışanlara odaklanarak erken eylem bütçesi ($3,000) yönlendirilir.
- **Modelin Sınırlılıkları:**
  - Özel kişisel sebepler (sağlık, ailevi) veya ani küresel ekonomik dalgalanmalar tarihsel İK verisinde yer almadığı için tahmin edilemeyebilir.
  - Panel veri yapısı nedeniyle modelin doğruluğu zamanla eskiyebilir; bu nedenle **yılda en az 1 kez yeni verilerle retraining** yapılması zorunludur.

---

# 🛠️ 4. Yöntem & CRISP-DM Metodolojisi

- **Business Understanding:** İstifa maliyetlerini minimize etmek ve en doğru tahminleme modelini tespit etmek.
- **Data Understanding (EDA):** Dengesiz sınıf dağılımı (%2.99 Churn) ve çoklu satır sızıntı riskleri analiz edildi.
- **Data Preparation:** Hedef sızıntıları giderildi, `GroupShuffleSplit` ile sızıntısız veri bölünmesi sağlandı, OneHot & scaling uygulandı.
- **Modeling:** 10 model eğitildi ve GroupKFold CV ile çapraz doğrulama yapısı kuruldu.

---

# 🏢 5. İş Analitiği & İstifa Davranışları (EDA)

- **Departman Bazlı Risk Dağılımı:**
  - Belirli birimlerde (örn. operasyonel birimler) istifa oranları genel ortalamanın üzerindedir.
  - Bu durum, İK'nın bütçesini tüm şirkete dağıtmak yerine riskli departmanlara odaklamasını gerektirir.
- **Kıdem Yılı (Tenure) Analizi:**
  - Çalışanlar özellikle ilk yıllarında ve belirli kariyer eşiklerinde (örn. 1. ve 5. yıl) daha yüksek istifa eğilimi göstermektedir.
  - Erken elde tutma programları bu kritik kıdem dönemlerine göre şekillendirilmelidir.

---

# 📊 6. Bulgular & 10 Modelin Yarıştırılması

| Model | Test Accuracy | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.987** | **0.618** | **0.749** | **0.932** |
| Random Forest | 0.975 | 0.729 | 0.644 | 0.912 |
| Hist Gradient Boosting | 0.971 | 0.850 | 0.640 | 0.935 |
| Decision Tree | 0.962 | 0.788 | 0.561 | 0.888 |
| AdaBoost | 0.980 | 0.366 | 0.535 | 0.896 |
| K-Nearest Neighbors | 0.976 | 0.369 | 0.487 | 0.837 |
| Logistic Regression | 0.869 | 0.758 | 0.262 | 0.880 |

- **Yarışma Sonucu:** Precision-Recall dengesini en iyi sağlayan ve en yüksek F1 puanını alan model **Gradient Boosting** olmuştur.

---

# 🛡️ 7. Model Savunması (Model Defense)

### Neden Gradient Boosting Kazandı?
- **Dengesiz Veri Dayanıklılığı:** Sınıf dengesizliğinde ağırlık dengesini iyi kurarak azınlık sınıfı (istifa edenler) en kararlı biçimde ayırt etti.
- **Ağaç Tabanlı Topluluk (Ensemble) Gücü:** Zayıf karar ağaçlarını ardışık olarak eğiterek hataları minimize eder ve overfitting farkını (yalnızca %0.01) en düşük seviyede tutar.
- **Optimizasyon Başarısı:** `RandomizedSearchCV` optimizasyonuyla en iyi hiperparametreler belirlendi:
  - `n_estimators`: 50, `max_depth`: 5, `learning_rate`: 0.05.

---

# 💰 8. Karar Eşiği Optimizasyonu (Threshold Tuning)

### Varsayılan Eşik (0.50) vs Optimal Eşik (0.10)
- **Varsayılan Eşik (0.50) Finansal Sonucu:**
  - TP: 170 | FP: 9 | FN: 136 (kaçırılan çalışan)
  - **Net Yıllık Tasarruf: $1,503,000**
- **Optimal Eşik (0.10) Finansal Sonucu:**
  - TP: 227 | FP: 57 | FN: 79 (kaçırılan çalışan 136'dan 79'a indi!)
  - **Net Yıllık Tasarruf: $1,872,000**
- **Optimizasyonun Katkısı:** Karar eşiği 0.10 yapılarak daha fazla istifa riski altındaki çalışan yakalanmış ve şirkete **$369,000 Ek Net Tasarruf** kazandırılmıştır!

---

# 🚀 9. Ürünleştirme & Canlı Portal

- **Nihai Kayıt:** `best_model.joblib` ve `pipeline.joblib` olarak modeller paketlendi.
- **Streamlit Portalı (`streamlit_app.py`):**
  - **Tekil Tahmin:** Çalışan özellikleri girilerek risk skoru ve İK aksiyon önerileri üretilir.
  - **Toplu Tahmin (CSV):** Toplu listeler taranarak yüksek riskli personeller önceliklendirilir.
  - **Model Dashboard:** Confusion Matrix, ROC ve İş Analitiği grafikleri canlı sunulur.

**Çalıştırma Komutu:**
```bash
streamlit run final-project/app/streamlit_app.py
```

---

# Teşekkürler

**Sorular & Tartışma**

*Proje Raporu: [report.pdf](file:///Users/ahmethaktanerd/Desktop/yenisonödev/final-project/report.pdf)*
