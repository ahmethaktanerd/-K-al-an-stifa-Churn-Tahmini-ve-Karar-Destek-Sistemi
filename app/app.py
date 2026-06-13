import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Sayfa yapılandırması
st.set_page_config(
    page_title="İK İstifa Risk Analizi Portalı",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Enjeksiyonu
def inject_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #F8FAFC;
        }
        .hero-card {
            background: linear-gradient(135deg, #2E86AB 0%, #1F5F7A 100%);
            color: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(46, 134, 171, 0.15);
            margin-bottom: 25px;
        }
        .metric-card {
            background-color: white;
            border: 1px solid #E2E8F0;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
            text-align: center;
        }
        .metric-val {
            font-size: 28px;
            font-weight: bold;
            color: #2E86AB;
        }
        .metric-label {
            font-size: 14px;
            color: #64748B;
            margin-top: 5px;
        }
        .result-safe {
            background-color: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-radius: 15px;
            padding: 25px;
            color: #065F46;
        }
        .result-danger {
            background-color: #FEF2F2;
            border: 1px solid #FEE2E2;
            border-radius: 15px;
            padding: 25px;
            color: #991B1B;
        }
        .advice-box {
            background-color: #FFFBEB;
            border-left: 5px solid #F59E0B;
            border-radius: 4px;
            padding: 15px;
            margin-top: 15px;
            color: #1E293B !important; /* Force dark text regardless of dark mode */
        }
        .advice-box h4, .advice-box b {
            color: #B45309 !important; /* Dark orange for header and bold text */
        }
        h1, h2, h3 {
            color: #1E293B;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar Menu Radio Button Styling (Rounded Boxes) */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
            background-color: #E2E8F0 !important; /* A bit darker gray */
            border: 1px solid #94A3B8 !important;
            border-radius: 12px !important;
            padding: 12px 15px !important;
            margin-bottom: 10px !important;
            cursor: pointer !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label p,
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label div {
            color: #1E293B !important; /* Black/Dark text */
        }
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
            background-color: #CBD5E1 !important; /* Darker gray on hover */
            border-color: #64748B !important;
        }
        /* Hide the actual radio circle */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        /* Style when selected */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
            background-color: #94A3B8 !important; /* Distinct darker gray for selected */
            border-color: #475569 !important;
        }
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) div,
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) p {
            color: #000000 !important; /* Solid black for selected text */
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# Proje kök dizinini ve alt yolları dinamik belirleme
base_dir = Path(".")
if not (base_dir / "models").exists() and (base_dir / "final-project").exists():
    base_dir = base_dir / "final-project"

PIPELINE_PATH = base_dir / "models/pipeline.joblib"
MODEL_PATH = base_dir / "models/best_model.joblib"
DATA_PATH = base_dir / "data/raw/veri_seti.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path("veri_seti.csv") # Fallback to workspace root data if final-project/data/raw/veri_seti.csv not present

SUMMARY_PATH = base_dir / "reports/summary.json"
COMPARISON_PATH = base_dir / "reports/model_comparison.csv"

# Caching Data Loading
@st.cache_data
def get_unique_values():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return {
            "city_name": sorted(df["city_name"].dropna().unique()),
            "department_name": sorted(df["department_name"].dropna().unique()),
            "job_title": sorted(df["job_title"].dropna().unique()),
            "BUSINESS_UNIT": sorted(df["BUSINESS_UNIT"].dropna().unique()),
            "store_name": sorted(df["store_name"].dropna().unique()),
            "dept_jobs": df.groupby('department_name')['job_title'].unique().apply(list).to_dict()
        }
    else:
        # Fallback values if dataset not found
        return {
            "city_name": ["Vancouver", "Terrace", "Kelowna", "Victoria"],
            "department_name": ["Executive", "Store Management", "Meats", "Produce"],
            "job_title": ["CEO", "Store Manager", "Cashier", "Produce Manager"],
            "BUSINESS_UNIT": ["STORES", "HEADOFFICE"],
            "store_name": [35, 18, 16, 37],
            "dept_jobs": {"Executive": ["CEO", "VP Stores"], "Store Management": ["Store Manager"], "Meats": ["Meats Manager", "Meat Cutter"], "Produce": ["Produce Manager", "Produce Clerk"]}
        }

@st.cache_resource
def load_ml_assets():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    pipeline = joblib.load(PIPELINE_PATH) if PIPELINE_PATH.exists() else None
    return model, pipeline

unique_vals = get_unique_values()
model, pipeline = load_ml_assets()

# İngilizce terimleri Türkçe göstermek için sözlük (Modelin orijinal veriyi alması korunur)
TR_MAPPING = {
    "gender": {"Female": "Kadın", "Male": "Erkek"},
    "business_unit": {"HEADOFFICE": "Merkez Ofis", "STORES": "Mağazalar"},
    "department": {
        "Executive": "Yönetim", "Store Management": "Mağaza Yönetimi", "Meats": "Et Ürünleri", "Recruitment": "İşe Alım",
        "Training": "Eğitim", "Labor Relations": "İşçi İlişkileri", "HR Technology": "İK Teknolojileri", 
        "Employee Records": "Çalışan Kayıtları", "Compensation": "Ücretlendirme", "Legal": "Hukuk", 
        "Produce": "Tarım Ürünleri", "Accounts Receiveable": "Alacak Hesapları", "Bakery": "Fırın", 
        "Information Technology": "Bilgi Teknolojileri", "Accounts Payable": "Borç Hesapları", "Audit": "Denetim", 
        "Accounting": "Muhasebe", "Investment": "Yatırım", "Dairy": "Süt Ürünleri", "Processed Foods": "İşlenmiş Gıdalar", 
        "Customer Service": "Müşteri Hizmetleri"
    },
    "job_title": {
        "CEO": "CEO", "VP Stores": "Mağazalar Başkan Yardımcısı", "Legal Counsel": "Hukuk Müşaviri",
        "VP Human Resources": "İK Başkan Yardımcısı", "VP Finance": "Finans Başkan Yardımcısı",
        "Exec Assistant, VP Stores": "Yönetici Asistanı, Mağazalar", "Exec Assistant, Legal Counsel": "Yönetici Asistanı, Hukuk",
        "CHief Information Officer": "Bilişim Başkanı (CIO)", "Store Manager": "Mağaza Müdürü", "Meats Manager": "Et Ürünleri Müdürü",
        "Exec Assistant, Human Resources": "Yönetici Asistanı, İK", "Exec Assistant, Finance": "Yönetici Asistanı, Finans",
        "Director, Recruitment": "Direktör, İşe Alım", "Director, Training": "Direktör, Eğitim",
        "Director, Labor Relations": "Direktör, İşçi İlişkileri", "Director, HR Technology": "Direktör, İK Teknolojileri",
        "Director, Employee Records": "Direktör, Kayıtlar", "Director, Compensation": "Direktör, Ücretlendirme",
        "Corporate Lawyer": "Şirket Avukatı", "Produce Manager": "Tarım Ürünleri Müdürü",
        "Director, Accounts Receivable": "Direktör, Alacak Hesapları", "Bakery Manager": "Fırın Müdürü",
        "Systems Analyst": "Sistem Analisti", "Director, Accounts Payable": "Direktör, Borç Hesapları",
        "Director, Audit": "Direktör, Denetim", "Director, Accounting": "Direktör, Muhasebe",
        "Director, Investments": "Direktör, Yatırımlar", "Dairy Person": "Süt Ürünleri Personeli",
        "Recruiter": "İşe Alım Uzmanı", "Processed Foods Manager": "İşlenmiş Gıdalar Müdürü",
        "Customer Service Manager": "Müşteri Hizmetleri Müdürü", "Trainer": "Eğitmen",
        "Meat Cutter": "Kasap", "Labor Relations Analyst": "İşçi İlişkileri Analisti",
        "Dairy Manager": "Süt Ürünleri Müdürü", "HRIS Analyst": "İK Sistemleri Analisti",
        "Benefits Admin": "Yan Haklar Yöneticisi", "Compensation Analyst": "Ücretlendirme Analisti",
        "Accounts Receiveable Clerk": "Alacak Hesapları Uzmanı", "Accounts Payable Clerk": "Borç Hesapları Uzmanı",
        "Baker": "Fırıncı", "Auditor": "Denetçi", "Accounting Clerk": "Muhasebe Uzmanı",
        "Investment Analyst": "Yatırım Analisti", "Produce Clerk": "Tarım Ürünleri Uzmanı",
        "Shelf Stocker": "Raf Görevlisi", "Cashier": "Kasiyer"
    }
}

# Sidebar Menüsü
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2>📊 İK Karar Destek</h2>
        <p style='color: #64748B;'>Çalışan İstifa Riski Tahmini</p>
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Menü Seçimi",
    [
        "🏠 Ana Sayfa",
        "👤 Tekil Çalışan Analizi",
        "📂 Toplu İstifa Analizi (CSV)",
        "📊 Model Performans Dashboard",
        "ℹ️ Yardım & Dokümantasyon",
        "ℹ️ Hakkımda"
    ]
)

# ----------------------------------------------------
# MENU 0: ANA SAYFA
# ----------------------------------------------------
if menu == "🏠 Ana Sayfa":
    st.markdown(
        """
        <style>
        .hero-container {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            padding: 70px 40px;
            border-radius: 24px;
            text-align: center;
            color: white;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.15);
            margin-bottom: 40px;
        }
        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, rgba(255,255,255,0) 50%);
            animation: pulse 8s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); opacity: 0.8; }
        }
        .hero-title {
            font-size: 46px !important;
            font-weight: 800 !important;
            margin-bottom: 20px !important;
            background: linear-gradient(to right, #38BDF8, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            position: relative;
            z-index: 1;
            line-height: 1.2 !important;
            margin-top: 0 !important;
        }
        .hero-subtitle {
            font-size: 18px !important;
            color: #94A3B8 !important;
            max-width: 650px;
            margin: 0 auto !important;
            line-height: 1.6 !important;
            position: relative;
            z-index: 1;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 20px;
            padding: 35px 25px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        }
        .glass-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.08);
            border-color: rgba(56, 189, 248, 0.3);
        }
        .glass-icon {
            font-size: 48px;
            margin-bottom: 20px;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
        }
        .glass-title {
            font-size: 20px !important;
            font-weight: 700 !important;
            color: #1E293B !important;
            margin-bottom: 12px !important;
            margin-top: 0 !important;
        }
        .glass-text {
            font-size: 15px !important;
            color: #64748B !important;
            line-height: 1.6 !important;
            margin-bottom: 0 !important;
        }
        </style>

        <div class="hero-container">
            <h1 class="hero-title">Akıllı İK Platformu</h1>
            <p class="hero-subtitle">
                Geleceğin insan kaynakları yönetimi burada. Makine öğrenmesi algoritmalarıyla çalışanlarınızın sadakatini analiz edin,
                kayıp risklerini önceden görün ve stratejik kararlarınızı veriyle güçlendirin.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-icon">🎯</div>
                <h3 class="glass-title">Bireysel Tahmin</h3>
                <p class="glass-text">Tek bir çalışanın metriklerini girerek anında yapay zeka tabanlı istifa risk skoru ve aksiyon önerisi alın.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-icon">🚀</div>
                <h3 class="glass-title">Toplu Analiz</h3>
                <p class="glass-text">Yüzlerce çalışanın verisini saniyeler içinde işleyin, riskli grupları keşfedin ve departman bütçenizi koruyun.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-icon">🧠</div>
                <h3 class="glass-title">Model Zekası</h3>
                <p class="glass-text">Algoritmaların başarı oranlarını, hata matrislerini ve personelin işten ayrılma sebeplerindeki en önemli faktörleri inceleyin.</p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br><br><p style='text-align:center; color:#94A3B8; font-size:15px; font-weight:500;'>🚀 Hemen başlamak için sol menüden bir işlem seçin.</p>", unsafe_allow_html=True)

# ----------------------------------------------------
# MENU 1: HAKKIMDA
# ----------------------------------------------------
elif menu == "ℹ️ Hakkımda":
    st.markdown(
        """
        <div class="hero-card">
            <h1>Çalışan İstifa (Churn) Tahmin Portalı</h1>
            <p>CRISP-DM Metodolojisiyle Geliştirilen İleri Seviye Makine Öğrenmesi Tabanlı Karar Destek Sistemi</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Rapor özetini yükle
    savings = 1_503_000
    opt_savings = 2_055_000
    opt_threshold = 0.10
    model_name = "Gradient Boosting"
    test_accuracy = "98.7%"
    
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            savings = summary.get("net_savings", savings)
            opt_savings = summary.get("optimal_net_savings", opt_savings)
            opt_threshold = summary.get("optimal_threshold", opt_threshold)
            model_name = summary.get("best_model_name", model_name)
            # Find metrics
            for metric in summary.get("metrics", []):
                if metric["Model"] == model_name:
                    test_accuracy = f"{metric['Test Accuracy'] * 100:.1f}%"
                    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">${opt_savings:,.2f}</div>
                <div class="metric-label">İK Optimum Net Tasarruf</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{opt_threshold:.2f}</div>
                <div class="metric-label">Optimum Karar Eşiği</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{model_name}</div>
                <div class="metric-label">Aktif En Başarılı Model</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{test_accuracy}</div>
                <div class="metric-label">Model Test Doğruluğu</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("### 🎯 Projenin İş Değeri ve Amaç")
    st.write(
        """
        Bir çalışanın istifa edip şirketten ayrılması; işe alım, onboarding süreçleri, zaman kaybı ve departman içi
        kültürel kayıplar dahil olmak üzere ortalama **15,000 USD** ikame maliyeti doğurmaktadır.
        
        Bu sistem, kritik personelin ayrılma olasılığını önceden tahmin ederek İK departmanına erken müdahale şansı tanır. 
        İK departmanı, yüksek riskli personeller için **3,000 USD** bütçeyle elde tutma aksiyonu (prim, terfi, rotasyon) alabilir ve 
        yapılan analizlere göre model destekli kararlar **milyon dolarlık işgücü kayıp maliyetini** önleyebilir.
        """
    )
    
    st.markdown("### 🛠️ CRISP-DM Çerçevesi")
    st.markdown(
        """
        *   **Business Understanding:** İstifa maliyetlerini düşürmek ve en doğru sınıflandırma modelini tespit etmek.
        *   **Data Understanding:** 49k+ panel veri kaydı incelendi, hedef sınıf dengesizliği (%3) ve panel verinin getirdiği sızıntı riskleri saptandı.
        *   **Data Preparation:** Eksik veriler ve hedef sızıntıları temizlendi, OneHot encoding yapıldı ve **çalışan ID bazlı GroupShuffleSplit** ile sızıntısız model verisi hazırlandı.
        *   **Modeling:** 10 farklı model eğitildi.
        *   **Evaluation:** F1-score, Recall, Overfitting ve cross-validation kararlılıkları ile finansal maliyet matrisi üzerinden modeller yarıştı.
        *   **Deployment:** Streamlit portalı yayına alındı.
        """
    )

# ----------------------------------------------------
# MENU 2: TEKİL ANALİZ
# ----------------------------------------------------
elif menu == "👤 Tekil Çalışan Analizi":
    st.markdown("## 👤 Tekil Çalışan İstifa Riski Sorgulama")
    st.write("Aşağıdaki formu doldurarak çalışanın özniteliklerine göre istifa risk skorunu hesaplayabilirsiniz.")
    
    if model is None or pipeline is None:
        st.error("⚠️ Model veya Preprocessing Pipeline dosyaları bulunamadı! Lütfen önce run_modeling.py scriptini çalıştırın.")
    else:
        # Form yapısı dinamik güncellemeleri engellediği için kaldırıldı.
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Yaş", min_value=18, max_value=70, value=35)
            length_of_service = st.number_input("Hizmet Süresi (Yıl)", min_value=0, max_value=30, value=5)
            gender = st.selectbox("Cinsiyet", ["Female", "Male"], format_func=lambda x: TR_MAPPING["gender"].get(x, x))
            
        with col2:
            city = st.selectbox("Şehir", unique_vals["city_name"])
            department = st.selectbox("Departman", unique_vals["department_name"], format_func=lambda x: TR_MAPPING["department"].get(x, x))
            
            # Dinamik Filtreleme: Seçilen departmana bağlı unvanları getir, yoksa hepsini göster
            valid_jobs = unique_vals.get("dept_jobs", {}).get(department, unique_vals["job_title"])
            job_title = st.selectbox("Unvan (Job Title)", sorted(valid_jobs), format_func=lambda x: TR_MAPPING["job_title"].get(x, x))
            
        with col3:
            business_unit = st.selectbox("İş Birimi (Business Unit)", unique_vals["BUSINESS_UNIT"], format_func=lambda x: TR_MAPPING["business_unit"].get(x, x))
            store_name = st.selectbox("Mağaza ID", unique_vals["store_name"])
            status_year = st.number_input("Değerlendirme Yılı", min_value=2006, max_value=2026, value=2015)
            
        submit_btn = st.button("Tahmin Et & Risk Analizi Yap", type="primary", use_container_width=True)
            
        if submit_btn:
            # Girdi verisi DataFrame oluşturma
            input_data = pd.DataFrame([{
                "age": age,
                "length_of_service": length_of_service,
                "city_name": city,
                "department_name": department,
                "job_title": job_title,
                "store_name": store_name,
                "gender_full": gender,
                "STATUS_YEAR": status_year,
                "BUSINESS_UNIT": business_unit
            }])
            
            # Pipeline ile preprocessing
            processed_input = pipeline.transform(input_data)
            
            # Predict
            pred = model.predict(processed_input)[0]
            
            # Predict Proba
            proba = 0.0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(processed_input)[0][1]
            
            st.markdown("### 📊 Analiz Sonucu")
            
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="result-danger">
                        <h3>⚠️ YÜKSEK RİSK: Çalışanın İstifa Etme Olasılığı Yüksek!</h3>
                        <p>Model Tahmin Olasılığı: <b>%{proba*100:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    """
                    <div class="advice-box">
                        <h4>💡 İK Önerilen Elde Tutma Eylemleri:</h4>
                        <ul>
                            <li><b>Birebir Görüşme:</b> Çalışanın motivasyonu, iş tatmini ve tükenmişlik seviyesi değerlendirilmelidir.</li>
                            <li><b>Kariyer Planlaması:</b> Çalışana şirket içi gelişim yolları ve eğitim fırsatları sunulmalıdır.</li>
                            <li><b>Finansal Teşvikler:</b> Piyasa standartları göz önüne alınarak prim veya ücret iyileştirmesi değerlendirilmelidir.</li>
                            <li><b>Esnek Çalışma:</b> İş-yaşam dengesini kurmak üzere çalışma koşulları esnetilebilir.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-safe">
                        <h3>✅ DÜŞÜK RİSK: Çalışanın Şirkette Kalma Eğilimi Yüksek.</h3>
                        <p>Model Tahmin Olasılığı (İstifa Riski): <b>%{proba*100:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ----------------------------------------------------
# MENU 3: TOPLU ANALİZ
# ----------------------------------------------------
elif menu == "📂 Toplu İstifa Analizi (CSV)":
    st.markdown("## 📂 Toplu İstifa Riski Analizi")
    st.write("Çalışan listesini içeren bir CSV dosyası yükleyerek tüm çalışanların istifa riskini aynı anda tahmin edebilirsiniz.")
    
    if model is None or pipeline is None:
        st.error("⚠️ Model veya Preprocessing Pipeline dosyaları bulunamadı! Lütfen önce run_modeling.py scriptini çalıştırın.")
    else:
        uploaded_file = st.file_uploader("Çalışan Veri Kümesi Yükleyin (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Yüklenen Veri Önizlemesi (İlk 5 Satır):")
            st.dataframe(input_df.head())
            
            # Kolon kontrolü
            required_cols = ['age', 'length_of_service', 'city_name', 'department_name', 'job_title', 'store_name', 'gender_full', 'STATUS_YEAR', 'BUSINESS_UNIT']
            missing_cols = [c for c in required_cols if c not in input_df.columns]
            
            if missing_cols:
                st.error(f"⚠️ Yüklenen dosyada bazı zorunlu sütunlar eksik: {missing_cols}")
            else:
                if st.button("Toplu Risk Hesapla"):
                    # Preprocessing
                    processed_data = pipeline.transform(input_df[required_cols])
                    
                    # Tahminler
                    preds = model.predict(processed_data)
                    probas = model.predict_proba(processed_data)[:, 1] if hasattr(model, "predict_proba") else preds
                    
                    output_df = input_df.copy()
                    output_df["İstifa Riski Tahmini"] = np.where(preds == 1, "Yüksek Risk", "Düşük Risk")
                    output_df["İstifa Olasılığı (%)"] = (probas * 100).round(2)
                    
                    st.success("Tüm çalışanların tahminleri başarıyla üretildi!")
                    
                    st.markdown("---")
                    
                    # 1. KPI Kartları (Detaylı Analiz)
                    high_risk_count = (preds == 1).sum()
                    total_count = len(preds)
                    avg_risk = output_df["İstifa Olasılığı (%)"].mean()
                    potential_cost = high_risk_count * 15000
                    
                    st.markdown("### 📊 Analiz Özeti ve Finansal Etki")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    with kpi1:
                        st.metric(label="Değerlendirilen Personel", value=f"{total_count} Kişi")
                    with kpi2:
                        st.metric(label="Yüksek Riskli Personel", value=f"{high_risk_count} Kişi", delta=f"%{(high_risk_count/total_count)*100:.1f} Risk Oranı", delta_color="inverse")
                    with kpi3:
                        st.metric(label="Ortalama İstifa Olasılığı", value=f"%{avg_risk:.1f}")
                    with kpi4:
                        st.metric(label="Önlenebilir Kayıp Maliyeti", value=f"${potential_cost:,.0f}", help="Riskli çalışanların ayrılması durumunda şirkete tahmini ikame maliyeti ($15,000/kişi)")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Sekmeler (Tabs) oluşturma
                    tab1, tab2, tab3 = st.tabs(["🎯 Genel Risk ve Demografi", "🏢 Departman ve Unvan Analizi", "🚨 Öncelikli Eylem Listesi"])
                    
                    with tab1:
                        col_pie, col_demo = st.columns(2)
                        with col_pie:
                            # Risk Seviyelerini detaylandırma
                            conditions = [
                                (output_df["İstifa Olasılığı (%)"] >= 80),
                                (output_df["İstifa Olasılığı (%)"] >= 50) & (output_df["İstifa Olasılığı (%)"] < 80),
                                (output_df["İstifa Olasılığı (%)"] >= 20) & (output_df["İstifa Olasılığı (%)"] < 50),
                                (output_df["İstifa Olasılığı (%)"] < 20)
                            ]
                            choices = ["Kritik Risk (>%80)", "Yüksek Risk (%50-80)", "Orta Risk (%20-50)", "Düşük Risk (<%20)"]
                            output_df["Risk Kategorisi"] = np.select(conditions, choices, default="Bilinmiyor")
                            
                            risk_cat_counts = output_df["Risk Kategorisi"].value_counts()
                            fig_risk = px.pie(
                                values=risk_cat_counts.values,
                                names=risk_cat_counts.index,
                                title="Risk Kategorisi Dağılımı",
                                color=risk_cat_counts.index,
                                color_discrete_map={
                                    "Kritik Risk (>%80)": "#7F1D1D", 
                                    "Yüksek Risk (%50-80)": "#DC2626",
                                    "Orta Risk (%20-50)": "#F59E0B",
                                    "Düşük Risk (<%20)": "#10B981"
                                }
                            )
                            fig_risk.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig_risk, use_container_width=True)
                            
                        with col_demo:
                            if high_risk_count > 0 and len(output_df) > high_risk_count:
                                high_risk_age = output_df[output_df["İstifa Riski Tahmini"] == "Yüksek Risk"]["age"].mean()
                                high_risk_srv = output_df[output_df["İstifa Riski Tahmini"] == "Yüksek Risk"]["length_of_service"].mean()
                                low_risk_age = output_df[output_df["İstifa Riski Tahmini"] == "Düşük Risk"]["age"].mean()
                                low_risk_srv = output_df[output_df["İstifa Riski Tahmini"] == "Düşük Risk"]["length_of_service"].mean()
                                
                                comp_df = pd.DataFrame({
                                    "Risk Grubu": ["Yüksek Risk", "Düşük Risk"],
                                    "Ortalama Yaş": [high_risk_age, low_risk_age],
                                    "Ortalama Hizmet Yılı": [high_risk_srv, low_risk_srv]
                                })
                                fig_comp = go.Figure(data=[
                                    go.Bar(name='Ortalama Yaş', x=comp_df['Risk Grubu'], y=comp_df['Ortalama Yaş'], marker_color='#3b82f6'),
                                    go.Bar(name='Ortalama Hizmet Yılı', x=comp_df['Risk Grubu'], y=comp_df['Ortalama Hizmet Yılı'], marker_color='#10b981')
                                ])
                                fig_comp.update_layout(barmode='group', title="Yaş ve Kıdem Karşılaştırması", margin=dict(t=40, b=0, l=0, r=0))
                                st.plotly_chart(fig_comp, use_container_width=True)
                            else:
                                st.info("Kıyaslama için yeterli çeşitlilikte risk grubu oluşmadı.")

                    with tab2:
                        col_dept, col_title = st.columns(2)
                        with col_dept:
                            dept_risk = output_df.groupby("department_name")["İstifa Olasılığı (%)"].mean().reset_index()
                            dept_risk = dept_risk.sort_values("İstifa Olasılığı (%)", ascending=False).head(10)
                            fig_dept = px.bar(
                                dept_risk, 
                                x="İstifa Olasılığı (%)", 
                                y="department_name", 
                                orientation='h',
                                title="En Riskli 10 Departman (Ortalama %)",
                                color="İstifa Olasılığı (%)",
                                color_continuous_scale="Reds"
                            )
                            fig_dept.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig_dept, use_container_width=True)
                            
                        with col_title:
                            title_risk = output_df.groupby("job_title")["İstifa Olasılığı (%)"].mean().reset_index()
                            title_risk = title_risk.sort_values("İstifa Olasılığı (%)", ascending=False).head(10)
                            fig_title = px.bar(
                                title_risk, 
                                x="İstifa Olasılığı (%)", 
                                y="job_title", 
                                orientation='h',
                                title="En Riskli 10 Unvan (Ortalama %)",
                                color="İstifa Olasılığı (%)",
                                color_continuous_scale="Oranges"
                            )
                            fig_title.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=40, b=0, l=0, r=0))
                            st.plotly_chart(fig_title, use_container_width=True)

                    with tab3:
                        st.markdown(f"### 🚨 Öncelikli Aksiyon Alınması Gereken Çalışanlar")
                        st.write("Aşağıdaki listede istifa riski yüksek/kritik olan çalışanlar sıralanmıştır. Tabloyu inceleyerek eylem planı oluşturabilirsiniz.")
                        high_risk_df = output_df[output_df["İstifa Riski Tahmini"] == "Yüksek Risk"].sort_values("İstifa Olasılığı (%)", ascending=False)
                        
                        if not high_risk_df.empty:
                            display_cols = ["age", "length_of_service", "department_name", "job_title", "city_name", "İstifa Olasılığı (%)", "Risk Kategorisi"]
                            st.dataframe(
                                high_risk_df[display_cols].style.background_gradient(
                                    subset=['İstifa Olasılığı (%)'], cmap='Reds'
                                ),
                                use_container_width=True
                            )
                        else:
                            st.success("Harika! Bu grupta yüksek istifa riski taşıyan kimse bulunamadı.")
                    
                    # CSV İndirme Butonu
                    csv_data = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Tahmin Sonuçlarını İndir (CSV)",
                        data=csv_data,
                        file_name="toplu_istifa_tahminleri_detayli.csv",
                        mime="text/csv"
                    )

# ----------------------------------------------------
# MENU 4: MODEL PERFORMANS
# ----------------------------------------------------
elif menu == "📊 Model Performans Dashboard":
    st.markdown("## 📊 Model Performans ve Karşılaştırma Panel")
    
    if not COMPARISON_PATH.exists():
        st.warning("⚠️ Karşılaştırma verileri bulunamadı. Lütfen önce run_modeling.py scriptini çalıştırın.")
    else:
        results_df = pd.read_csv(COMPARISON_PATH)
        
        # Grafik 1: F1-Score Karşılaştırması
        fig_f1 = px.bar(
            results_df.sort_values("F1-Score", ascending=False),
            x="Model", y="F1-Score", color="Recall",
            title="10 Modelin F1-Score ve Recall Başarı Kıyaslaması",
            color_continuous_scale="Purples",
            text="F1-Score"
        )
        fig_f1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_f1, use_container_width=True)
        
        # Grafik 2: Overfit Analizi (Train Accuracy vs Test Accuracy)
        results_df["Overfit"] = results_df["Train Accuracy"] - results_df["Test Accuracy"]
        fig_overfit = px.bar(
            results_df.sort_values("Overfit", ascending=True),
            x="Model", y="Overfit", color="Overfit",
            title="Modellerin Overfitting Seviyesi (Train vs Test Farkı - Düşük Olan Daha Güvenlidir)",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_overfit, use_container_width=True)
        
        # Tablo Görünümü
        st.markdown("### Tüm Modellerin Detaylı Performans Tablosu")
        st.dataframe(results_df)
        
        # Kaydedilen görselleri yükleme
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🎯 Seçilen En İyi Modelin Hata Matrisi (Confusion Matrix)")
            cm_path = base_dir / "figures/confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Karmaşıklık Matrisi")
                
            st.markdown("### 🔑 Öznitelik Önem Düzeyleri (Feature Importance)")
            feat_path = base_dir / "figures/feature_importance.png"
            if feat_path.exists():
                st.image(str(feat_path), caption="Top 10 En Önemli Öznitelik")

        with col_right:
            st.markdown("### 📈 Seçilen En İyi Modelin ROC Eğrisi ve AUC Değeri")
            roc_path = base_dir / "figures/roc_curve.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Eğrisi")
                
            st.markdown("### 📊 Hassasiyet-Duyarlılık (Precision-Recall) Eğrisi")
            pr_path = base_dir / "figures/precision_recall_curve.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Eğrisi")

        st.markdown("### 🎯 Karar Eşiği Optimizasyonu (Threshold Tuning)")
        th_path = base_dir / "figures/threshold_tuning.png"
        if th_path.exists():
            st.image(str(th_path), caption="Eşik Olasılığı vs İK Bütçe Tasarrufu ($) - Optimum: 0.10")

        st.markdown("### 📈 İş Analitiği ve İstifa Davranışları")
        col_left_2, col_right_2 = st.columns(2)
        with col_left_2:
            st.markdown("#### 🏢 Departman Bazlı İstifa Oranları")
            dept_path = base_dir / "figures/department_churn.png"
            if dept_path.exists():
                st.image(str(dept_path), caption="Departman Bazlı İstifa Oranları (%)")
                
        with col_right_2:
            st.markdown("#### ⏳ Kıdem Yılına Göre İstifa Oranları")
            service_path = base_dir / "figures/service_length_churn.png"
            if service_path.exists():
                st.image(str(service_path), caption="Kıdem Yılına Göre İstifa Oranları (%)")

        st.markdown("### 🔍 Veri Kalitesi & İkili İlişkiler: Yaş vs Kıdem ve İstifa Durumu")
        biv_path = base_dir / "figures/bivariate_scatter.png"
        if biv_path.exists():
            st.image(str(biv_path), caption="Yaş vs Hizmet Süresi Dağılımı (Kırmızı Noktalar İstifa Edenleri Gösterir)")

# ----------------------------------------------------
# MENU 5: YARDIM & DOKÜMANTASYON
# ----------------------------------------------------
else:
    st.markdown("## ℹ️ Yardım & Kullanım Dokümantasyonu")
    st.markdown(
        """
        ### 💡 Uygulama Nasıl Kullanılır?
        1.  **Yönetici Özeti:** Projenin iş vizyonu, İK departmanına kazandırdığı bütçe tasarrufları ve CRISP-DM süreç detayları bu ekranda bulunur.
        2.  **Tekil Çalışan Analizi:** Belirli bir personelin yaş, kıdem, unvan, departman vb. özelliklerini form üzerinden girerek risk skorunu anında hesaplayabilirsiniz.
        3.  **Toplu İstifa Analizi:** Şirketteki tüm çalışanların toplu özniteliklerini içeren bir CSV yükleyip, istifa risk oranlarıyla birlikte listeyi güncelleyebilirsiniz. Tahmin edilen listeyi tekrar CSV formatında indirebilirsiniz.
        4.  **Model Performans Dashboard:** Hangi algoritmaların (10 farklı model) denendiği, başarı metrikleri (Accuracy, Recall, Precision, F1-Score) ve hata matrisleri bu sayfada detaylandırılır.
        
        ### 📋 CSV Dosya Yükleme Formatı (Kolon İsimleri):
        Toplu analiz yükleme dosyasında aşağıdaki kolonlar bulunmalıdır:
        *   `age`: Çalışanın yaşı (örn. 45)
        *   `length_of_service`: Çalışanın şirketteki kıdem yılı (örn. 12)
        *   `city_name`: Şehir adı (örn. Vancouver)
        *   `department_name`: Departman adı (örn. Meats)
        *   `job_title`: Çalışanın unvanı (örn. Meats Manager)
        *   `store_name`: Mağaza numarası (örn. 35)
        *   `gender_full`: Cinsiyet (Female/Male)
        *   `STATUS_YEAR`: Değerlendirme yapılan yıl (örn. 2015)
        *   `BUSINESS_UNIT`: STORES veya HEADOFFICE
        
        *Not: Modelin eğitimi `veri_seti.csv` üzerindeki panel veri yapısına dayanmaktadır. Sızıntıları engellemek için tüm preprocessing adımları pipeline içinde kapsüllenmiştir.*
        """
    )
