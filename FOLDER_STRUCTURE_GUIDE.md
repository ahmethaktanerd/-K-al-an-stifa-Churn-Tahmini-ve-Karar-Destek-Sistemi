# 📁 FINAL PROJECT - PROFESYONEL KLASÖR YAPISI

Tüm çalışmalar ve agent tanımları bu profesyonel klasör yapısını temel almalıdır.

## 🎯 Klasör Yapısı

```
final-project/
├── data/
│   ├── raw/                      # Orijinal veri seti (değiştirilmez)
│   │   └── veri_seti.csv
│   └── processed/                # Temizlenmiş veri
│       └── veri_seti_cleaned.csv
│
├── notebooks/
│   └── final_analysis.ipynb      # CRISP-DM analizi (EDA, Prep, Modelleme)
│
├── models/
│   ├── best_model.joblib         # En iyi seçilen nihai model
│   └── pipeline.joblib           # Preprocessing ve feature engineering pipeline
│
├── app/
│   ├── streamlit_app.py          # Streamlit kullanıcı arayüzü
│   └── api.py                    # FastAPI servisi (opsiyonel)
│
├── figures/                      # Grafik çıktıları (HTML, PNG)
│   ├── correlation_matrix.png
│   ├── missing_values.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── scripts/                      # Yardımcı Python scriptleri
│   └── generate_pdf_report.py    # PDF Rapor oluşturucu script
│
├── requirements.txt              # Bağımlı kütüphaneler listesi
├── README.md                     # Proje açıklaması ve Mermaid akışı
├── presentation.md               # Proje sunum slaytları (Marp formatında)
└── report.pdf                    # Final danışmanlık raporu (PDF)
```

---

## 📋 Agent'lerin Dosya Yolu Kullanım Kuralları

Her zaman `final-project/` dizini altında çalışırken relative path kullanımı:

### 🔍 EDA Aşaması
```python
# Ham veriyi oku
df = pd.read_csv('../data/raw/veri_seti.csv')

# Temizlenmiş/işlenmiş veriyi kaydet
df.to_csv('../data/processed/veri_seti_cleaned.csv', index=False)
```

### 🛠️ Preprocessing ve Modelleme Aşaması
```python
# Pipeline ve Model Kaydetme
import joblib
joblib.dump(pipeline, '../models/pipeline.joblib')
joblib.dump(best_model, '../models/best_model.joblib')
```

### 🚀 Streamlit Deployment Aşaması
```python
# Model ve Pipeline Yükleme
import joblib
pipeline = joblib.load('models/pipeline.joblib')
model = joblib.load('models/best_model.joblib')
```
