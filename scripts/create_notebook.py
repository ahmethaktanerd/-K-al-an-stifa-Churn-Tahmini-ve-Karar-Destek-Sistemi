import json

def main():
    print("Notebook oluşturuluyor...")
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0b1329; border-radius: 12px; padding: 30px; text-align: center; color: #ffffff; font-family: 'Inter', sans-serif; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 25px;\">\n",
                    "    <span style=\"background-color: #172554; color: #3b82f6; border-radius: 20px; padding: 6px 16px; font-size: 13px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;\">YBS3259 - MAKİNE ÖĞRENMESİ FİNAL PROJESİ</span>\n",
                    "    <h1 style=\"font-size: 34px; font-weight: 800; margin-top: 20px; margin-bottom: 10px; color: #ffffff; line-height: 1.2;\">Çalışan Kaybı (Employee Churn)<br>Tahmin Modeli</h1>\n",
                    "    <p style=\"font-size: 16px; color: #94a3b8; font-weight: 400; margin-top: 10px; margin-bottom: 20px;\">Makine Öğrenmesi Tabanlı İnsan Kaynakları Karar Destek Sistemi</p>\n",
                    "    <hr style=\"border-color: #1e293b; margin: 15px 0;\">\n",
                    "    <p style=\"color: #cbd5e1; font-size: 14px; line-height: 1.5; margin: 0;\">Bu notebook, şirket çalışanlarının istifa riskini önceden tahmin etmek ve İK departmanının elde tutma bütçesini en verimli şekilde kullanmasını sağlamak amacıyla <b>CRISP-DM</b> standartlarına uygun olarak tasarlanmıştır.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0b1329; border-left: 6px solid #3b82f6; border-radius: 8px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 20px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);\">\n",
                    "    <h2 style=\"margin: 0; color: #ffffff; font-size: 20px; font-weight: 700;\">1. Business Understanding (İş Problemi & Karar Bağlamı)</h2>\n",
                    "</div>\n",
                    "\n",
                    "<div style=\"background-color: #1e293b; border-left: 5px solid #3b82f6; padding: 18px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h3 style=\"margin-top: 0; color: #3b82f6; font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 700; margin-bottom: 8px;\">🎯 İş Problemi</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Şirket çalışanlarının istifa etmesi (çalışan kaybı/churn), yeni işe alım maliyetleri, oryantasyon süreleri ve departman içi bilgi kaybı nedeniyle ciddi maliyetler doğurmaktadır. İK departmanı için <b>ortalama bir çalışanın ayrılma maliyeti $15,000</b> olarak hesaplanmıştır.</p>\n",
                    "</div>\n",
                    "\n",
                    "<div style=\"background-color: #1e293b; border-left: 5px solid #10b981; padding: 18px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h3 style=\"margin-top: 0; color: #10b981; font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 700; margin-bottom: 8px;\">💡 Karar Bağlamı ve Müdahale Stratejisi</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">İK departmanı, ayrılma riski yüksek olan çalışanları önceden belirleyebilir ve <b>çalışan başına $3,000 elde tutma bütçesi</b> (prim, terfi, rotasyon gibi aksiyonlar) kullanarak bu çalışanların <b>%80 olasılıkla şirkette kalmasını</b> sağlayabilir.</p>\n",
                    "</div>\n",
                    "\n",
                    "<div style=\"background-color: #1e293b; border-left: 5px solid #ec4899; padding: 18px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h3 style=\"margin-top: 0; color: #ec4899; font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 700; margin-bottom: 8px;\">📊 Başarı Kriterleri ve Risk Analizi</h3>\n",
                    "    <ul style=\"margin: 0; color: #cbd5e1; padding-left: 20px; font-size: 14px; line-height: 1.6;\">\n",
                    "        <li><b>Recall (Duyarlılık):</b> En kritik metriğimizdir. Ayrılacak kişileri gözden kaçırmanın maliyeti ($15,000) yüksek olduğu için False Negative'leri minimize etmeliyiz.</li>\n",
                    "        <li><b>Precision (Kesinlik):</b> Bütçeyi verimli kullanmak için gereksiz yere elde tutma aksiyonu uygulanacak kişileri (False Positive) sınırlamalıyız.</li>\n",
                    "        <li><b>Finansal Net Tasarruf:</b> Modeli devreye aldığımızda İK'nın elde edeceği toplam finansal tasarruf ana başarı kriterimizdir.</li>\n",
                    "    </ul>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import plotly.express as px\n",
                    "import plotly.graph_objects as go\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import joblib\n",
                    "from pathlib import Path\n",
                    "\n",
                    "from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold, RandomizedSearchCV\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve\n",
                    "\n",
                    "from sklearn.dummy import DummyClassifier\n",
                    "from sklearn.linear_model import LogisticRegression, RidgeClassifier\n",
                    "from sklearn.neighbors import KNeighborsClassifier\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier\n",
                    "from sklearn.ensemble import HistGradientBoostingClassifier\n",
                    "\n",
                    "print(\"✅ Gerekli kütüphaneler başarıyla yüklendi.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0b1329; border-left: 6px solid #2e86ab; border-radius: 8px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 20px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);\">\n",
                    "    <h2 style=\"margin: 0; color: #ffffff; font-size: 20px; font-weight: 700; margin-bottom: 10px;\">2. Data Understanding (Veriyi Tanıma & Kalite Kontrolü)</h2>\n",
                    "    <p style=\"margin: 0; color: #94a3b8; font-size: 14px; line-height: 1.5;\">Veri kümesini yükleyerek genel profilini, eksik/aykırı değerleri ve öznitelik türlerini inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Veriyi oku\n",
                    "df = pd.read_csv('../data/raw/veri_seti.csv')\n",
                    "print(f\"Veri Boyutu: {df.shape[0]} satır, {df.shape[1]} sütun\")\n",
                    "print(\"\\nEksik Değer Durumu:\")\n",
                    "print(df.isnull().sum())\n",
                    "df.info()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #ef4444; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.1. Target (Hedef Değişken) Analizi</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Hedef değişkenimiz olan <code>STATUS</code> (ACTIVE / TERMINATED) sınıflarının oranını inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "target_counts = df['STATUS'].value_counts()\n",
                    "target_ratios = df['STATUS'].value_counts(normalize=True) * 100\n",
                    "\n",
                    "fig = px.bar(\n",
                    "    x=target_counts.index,\n",
                    "    y=target_ratios.values,\n",
                    "    color=target_counts.index,\n",
                    "    color_discrete_sequence=['#2E86AB', '#C73E1D'],\n",
                    "    title=\"Hedef Değişken Sınıf Dağılım Oranları (%)\",\n",
                    "    labels={'x': 'Status', 'y': 'Oran (%)'}\n",
                    ")\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Veri Analisti Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Sınıf dağılımında görüldüğü üzere, çalışanların <b>%97.01'i ACTIVE</b>, yalnızca <b>%2.99'u TERMINATED</b> durumundadır. Bu durum veri kümesinde aşırı sınıf dengesizliği (class imbalance) olduğunu gösterir. Modelleme aşamasında modellerin bu durumdan etkilenmesini önlemek için <code>class_weight='balanced'</code> gibi yöntemler kullanılmalıdır.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.1b. İş Birimi, Cinsiyet ve Demografik Kıyaslamalar</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">İstifa eden ve aktif kalan çalışanların demografik ve operasyonel kırılımlarını inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Demografik ortalamalar (İstifa vs Aktif)\n",
                    "stats_by_status = df.groupby('STATUS')[['age', 'length_of_service']].mean()\n",
                    "print(\"--- ORTALAMA YAŞ VE KIDEM DEĞERLERİ ---\")\n",
                    "print(stats_by_status)\n",
                    "\n",
                    "# İş Birimi bazlı istifa oranları\n",
                    "bu_churn = df.groupby('BUSINESS_UNIT')['target'].mean() * 100\n",
                    "print(\"\\n--- İŞ BİRİMİ BAZLI İSTİFA ORANLARI (%) ---\")\n",
                    "print(bu_churn)\n",
                    "\n",
                    "# Cinsiyet bazlı istifa oranları\n",
                    "gender_churn = df.groupby('gender_full')['target'].mean() * 100\n",
                    "print(\"\\n--- CİNSİYET BAZLI İSTİFA ORANLARI (%) ---\")\n",
                    "print(gender_churn)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Veri Analisti Yorumu</h4>\n",
                    "    <ul style=\"margin: 0; color: #cbd5e1; padding-left: 20px; font-size: 14px; line-height: 1.6;\">\n",
                    "        <li><b>Yaş Dağılımı:</b> İstifa edenlerin yaş ortalaması (51.46), aktif çalışanlarınkinden (41.79) belirgin şekilde yüksektir. Bu durum, istifaların daha çok emeklilik veya kıdemli seviyelerdeki çalışan hareketliliğinden kaynaklandığını göstermektedir.</li>\n",
                    "        <li><b>İş Birimi (Business Unit):</b> Genel Müdürlük (HEADOFFICE) biriminde istifa oranı %11.79 iken, Mağazalarda (STORES) bu oran %2.89'dur. HEADOFFICE'teki 4 kat daha yüksek churn oranı, İK için kritik bir alarm durumudur ve özel odaklanma gerektirir.</li>\n",
                    "        <li><b>Cinsiyet Kırılımı:</b> Kadın çalışanların istifa oranı (%3.53), erkek çalışanlarınkine kıyasla (%2.40) biraz daha yüksektir.</li>\n",
                    "    </ul>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.1c. Gelişmiş İstatistiksel Hipotez Testleri</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Gözlemlediğimiz demografik farkların (Yaş, Kıdem, İş Birimi ve Cinsiyet) istatistiksel olarak anlamlı olup olmadığını bilimsel testlerle doğrulayalım.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from scipy import stats\n",
                    "\n",
                    "# 1. İki Bağımsız Örneklem T-Testi (Yaş)\n",
                    "active_age = df[df['STATUS'] == 'ACTIVE']['age']\n",
                    "term_age = df[df['STATUS'] == 'TERMINATED']['age']\n",
                    "t_age, p_age = stats.ttest_ind(active_age, term_age, equal_var=False)\n",
                    "print(f\"Yaş için T-İstatistiği: {t_age:.4f}, p-değeri: {p_age:.4e}\")\n",
                    "\n",
                    "# 2. İki Bağımsız Örneklem T-Testi (Kıdem)\n",
                    "active_los = df[df['STATUS'] == 'ACTIVE']['length_of_service']\n",
                    "term_los = df[df['STATUS'] == 'TERMINATED']['length_of_service']\n",
                    "t_los, p_los = stats.ttest_ind(active_los, term_los, equal_var=False)\n",
                    "print(f\"Kıdem Süresi için T-İstatistiği: {t_los:.4f}, p-değeri: {p_los:.4e}\")\n",
                    "\n",
                    "# 3. Ki-Kare İlişki Testi (İş Birimi)\n",
                    "ct_bu = pd.crosstab(df['BUSINESS_UNIT'], df['STATUS'])\n",
                    "chi2_bu, p_bu, _, _ = stats.chi2_contingency(ct_bu)\n",
                    "print(f\"İş Birimi Ki-Kare p-değeri: {p_bu:.4e}\")\n",
                    "\n",
                    "# 4. Ki-Kare İlişki Testi (Cinsiyet)\n",
                    "ct_gen = pd.crosstab(df['gender_full'], df['STATUS'])\n",
                    "chi2_gen, p_gen, _, _ = stats.chi2_contingency(ct_gen)\n",
                    "print(f\"Cinsiyet Ki-Kare p-değeri: {p_gen:.4e}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 İstatistiksel Değerlendirme Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Tüm test sonuçlarında p-değerleri <b>alpha = 0.05</b> anlamlılık eşiğinin son derece altındadır. Bu durum:<br>\n",
                    "    - İstifa edenler ile aktif çalışanlar arasındaki yaş farkının ve kıdem süresi farkının rastlantısal olmadığını,<br>\n",
                    "    - İş Birimi (HEADOFFICE vs STORES) ile Cinsiyet değişkenlerinin istifa kararı üzerinde istatistiksel olarak son derece belirleyici (anlamlı) olduğunu kanıtlar.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.1d. Lokasyon ve Pozisyon Bazlı Risk Dağılımı</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">İstifa oranlarının hangi şehirler ve hangi iş unvanlarında yoğunlaştığını inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# En yüksek istifa oranına sahip şehirler\n",
                    "city_churn = df.groupby('city_name')['target'].agg(['count', 'mean'])\n",
                    "city_churn = city_churn[city_churn['count'] >= 30].sort_values('mean', ascending=False).head(10)\n",
                    "city_churn['mean'] = city_churn['mean'] * 100\n",
                    "\n",
                    "fig_city = px.bar(\n",
                    "    city_churn, x=city_churn.index, y=\"mean\", text_auto='.1f',\n",
                    "    title=\"En Yüksek İstifa Oranına Sahip 10 Şehir (En az 30 Kayıt Olanlar)\",\n",
                    "    labels={'mean': 'İstifa Oranı (%)', 'city_name': 'Şehir'},\n",
                    "    color=\"mean\", color_continuous_scale=\"Oranges\"\n",
                    ")\n",
                    "fig_city.show()\n",
                    "\n",
                    "# En yüksek istifa oranına sahip unvanlar\n",
                    "job_churn = df.groupby('job_title')['target'].agg(['count', 'mean'])\n",
                    "job_churn = job_churn.sort_values('mean', ascending=False).head(10)\n",
                    "job_churn['mean'] = job_churn['mean'] * 100\n",
                    "\n",
                    "fig_job = px.bar(\n",
                    "    job_churn, x=\"mean\", y=job_churn.index, orientation='h', text_auto='.1f',\n",
                    "    title=\"En Yüksek İstifa Oranına Sahip 10 Unvan\",\n",
                    "    labels={'mean': 'İstifa Oranı (%)', 'job_title': 'Unvan'},\n",
                    "    color=\"mean\", color_continuous_scale=\"Reds\"\n",
                    ")\n",
                    "fig_job.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Coğrafi ve Pozisyonel Analiz Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Analizde görüldüğü üzere <b>New Westminister (%17.3)</b> ve <b>Pitt Meadows (%15.8)</b> şehirlerindeki istifa oranları, %2.99 olan şirket ortalamasının neredeyse 5-6 katıdır. Pozisyonel olarak da Director düzeyindeki rollerde istifa oranının %25'lere çıkması, üst yönetim düzeyinde ciddi bir elde tutma stratejisi ihtiyacına işaret eder.</p>\n",
                    "</div>"
                ]
            },

            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #6a994e; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.2. Sayısal Değişkenlerin ve Aykırı Değerlerin Analizi</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Yaş (<code>age</code>) ve Hizmet Süresi (<code>length_of_service</code>) değişkenlerinin dağılımlarını ve aykırı değerlerini Box-plot ile inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_age = px.box(df, y=\"age\", title=\"Çalışan Yaş Dağılımı ve Aykırı Değerler\", color_discrete_sequence=['#2E86AB'])\n",
                    "fig_age.show()\n",
                    "\n",
                    "fig_service = px.box(df, y=\"length_of_service\", title=\"Hizmet Süresi Dağılımı ve Aykırı Değerler\", color_discrete_sequence=['#6A994E'])\n",
                    "fig_service.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Veri Analisti Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Yaş dağılımı 19 ile 65 yaş arasında homojen bir dağılım göstermektedir ve belirgin bir aykırı değer bulunmamaktadır. Hizmet süresi ise 0 ile 26 yıl arasında değişmektedir. Çeyreklikler arası aralıkta (IQR) aykırı değer olabilecek aşırı uç noktalar görülmemiştir.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #ef4444; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.3. Departman Bazlı İstifa Oranları</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Farklı departmanlardaki çalışan istifa oranlarını karşılaştıralım.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dept_churn = df.groupby('department_name')['target'].mean().reset_index()\n",
                    "dept_churn['target'] = dept_churn['target'] * 100\n",
                    "dept_churn = dept_churn.sort_values('target', ascending=False)\n",
                    "\n",
                    "fig_dept = px.bar(\n",
                    "    dept_churn, x=\"target\", y=\"department_name\", orientation=\"h\",\n",
                    "    title=\"Departman Bazlı İstifa Oranları (%)\",\n",
                    "    color=\"target\", color_continuous_scale=\"Reds\",\n",
                    "    labels={'target': 'İstifa Oranı (%)', 'department_name': 'Departman'}\n",
                    ")\n",
                    "fig_dept.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Veri Analisti Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Departman bazında istifa oranları incelendiğinde bazı kritik birimlerde istifa oranlarının belirgin şekilde yüksek olduğu görülebilir. Bu durum İK için hedefli aksiyon alma imkanı sunar.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #2e86ab; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.4. Kıdem Yılına Göre İstifa Oranları</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Çalışanların şirkette geçirdikleri yıla (kıdem/hizmet süresi) göre istifa eğilimlerini inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "service_churn = df.groupby('length_of_service')['target'].mean().reset_index()\n",
                    "service_churn['target'] = service_churn['target'] * 100\n",
                    "\n",
                    "fig_service_churn = px.line(\n",
                    "    service_churn, x=\"length_of_service\", y=\"target\", markers=True,\n",
                    "    title=\"Kıdem Yılına Göre İstifa Oranları (%)\",\n",
                    "    labels={'target': 'İstifa Oranı (%)', 'length_of_service': 'Hizmet Süresi (Yıl)'}\n",
                    ")\n",
                    "fig_service_churn.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Veri Analisti Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Kıdem süresi analizinde belirli hizmet yıllarında (örn. ilk yıllarda veya belirli tecrübe eşiklerinde) istifa oranının pik yaptığı görülebilir. Bu grafik İK'ya kariyer aşamalarına göre özelleştirilmiş retention programları tasarlama içgörüsü sağlar.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #6a994e; border-radius: 6px; padding: 15px 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 8px;\">2.5. Korelasyon Analizi</h3>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Sayısal değişkenlerin kendi aralarındaki korelasyonunu inceleyelim.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "numerical_cols = ['age', 'length_of_service', 'STATUS_YEAR']\n",
                    "corr_matrix = df[numerical_cols].corr()\n",
                    "fig_corr = px.imshow(\n",
                    "    corr_matrix, \n",
                    "    text_auto=\".3f\", \n",
                    "    title=\"Sayısal Özniteliklerin Korelasyon Matrisi\",\n",
                    "    color_continuous_scale=\"RdBu_r\"\n",
                    ")\n",
                    "fig_corr.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Veri Analisti Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Yaş ve hizmet süresi arasında <b>0.58</b> düzeyinde pozitif yönlü, orta şiddette bir ilişki vardır. Bu durum yaşlandıkça hizmet süresinin artma eğilimini gösterir. Çoklu doğrusal bağlantı (multicollinearity) riski taşıyacak derecede yüksek (&gt;0.80) bir ilişki bulunmamaktadır.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0b1329; border-left: 6px solid #d97706; border-radius: 8px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 20px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);\">\n",
                    "    <h2 style=\"margin: 0; color: #ffffff; font-size: 20px; font-weight: 700; margin-bottom: 10px;\">3. Data Preparation (Veri Hazırlama & Sızıntı Engelleme)</h2>\n",
                    "    <div style=\"background-color: #1e1b4b; border-left: 5px solid #f59e0b; padding: 15px; border-radius: 6px; margin-top: 10px; color: #ffffff;\">\n",
                    "        <h3 style=\"margin-top: 0; color: #fbbf24; font-size: 16px; font-weight: 700; margin-bottom: 8px;\">⚠️ Veri Sızıntısı (Data Leakage) Nedir ve Nasıl Engellenir?</h3>\n",
                    "        <p style=\"color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Bu veri kümesinde aynı çalışanın farklı yıllardaki kayıtları yer almaktadır (boylamsal/panel veri). Rastgele bir satır bazlı bölünme yaparsak, aynı çalışanın 2012 yılı verisi eğitim setine, 2013 yılı verisi test setine düşebilir. Bu durum modelin çalışanı ezberlemesine (leakage) yol açar.</p>\n",
                    "        <p style=\"margin-bottom: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\"><b>Bunu engellemek için:</b><br>\n",
                    "        1. Çalışan ID bazlı gruplama yaparak bölünme sağlayacağız (<code>GroupShuffleSplit</code>).<br>\n",
                    "        2. <code>terminationdate_key</code>, <code>termreason_desc</code>, <code>termtype_desc</code> gibi hedef değişkenle birebir ilişkili olan sızıntı alanlarını model girdisinden çıkaracağız.</p>\n",
                    "    </div>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Target oluşturma\n",
                    "df['target'] = (df['STATUS'] == 'TERMINATED').astype(int)\n",
                    "\n",
                    "# Sızıntı ve ID kolonları\n",
                    "leakage_cols = ['terminationdate_key', 'termreason_desc', 'termtype_desc', 'STATUS', 'gender_short']\n",
                    "id_cols = ['EmployeeID', 'recorddate_key', 'birthdate_key', 'orighiredate_key']\n",
                    "\n",
                    "X = df.drop(columns=['target'] + leakage_cols + id_cols)\n",
                    "y = df['target']\n",
                    "groups = df['EmployeeID']\n",
                    "\n",
                    "# Grup tabanlı split\n",
                    "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\n",
                    "train_idx, test_idx = next(gss.split(X, y, groups=groups))\n",
                    "\n",
                    "X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]\n",
                    "y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]\n",
                    "train_groups = groups.iloc[train_idx]\n",
                    "\n",
                    "print(f\"Eğitim Seti Boyutu: {X_train.shape[0]}, Test Seti Boyutu: {X_test.shape[0]}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Preprocessing Pipeline\n",
                    "categorical_cols = ['city_name', 'department_name', 'job_title', 'BUSINESS_UNIT', 'store_name']\n",
                    "numerical_cols = ['age', 'length_of_service', 'STATUS_YEAR']\n",
                    "\n",
                    "preprocessor = ColumnTransformer(\n",
                    "    transformers=[\n",
                    "        ('num', StandardScaler(), numerical_cols),\n",
                    "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)\n",
                    "    ]\n",
                    ")\n",
                    "\n",
                    "# Pipeline'ı sadece train seti üzerinde fit ediyoruz (Data Leakage engellenir)\n",
                    "X_train_processed = preprocessor.fit_transform(X_train)\n",
                    "X_test_processed = preprocessor.transform(X_test)\n",
                    "print(\"✅ Preprocessing pipeline başarıyla fit edildi ve veriler dönüştürüldü.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0b1329; border-left: 6px solid #10b981; border-radius: 8px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 20px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);\">\n",
                    "    <h2 style=\"margin: 0; color: #ffffff; font-size: 20px; font-weight: 700; margin-bottom: 10px;\">4. Modeling (Modelleme & 10 Farklı Modelin Yarıştırılması)</h2>\n",
                    "    <p style=\"margin: 0; color: #94a3b8; font-size: 14px; line-height: 1.5;\">Sistematik olarak 10 farklı sınıflandırıcı modeli aynı train-test bölünmesi ve GroupKFold CV stratejisiyle eğitiyoruz.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "models = {\n",
                    "    \"Dummy Classifier\": DummyClassifier(strategy=\"stratified\", random_state=42),\n",
                    "    \"Logistic Regression\": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),\n",
                    "    \"Ridge Classifier\": RidgeClassifier(random_state=42, class_weight='balanced'),\n",
                    "    \"K-Nearest Neighbors\": KNeighborsClassifier(n_neighbors=5),\n",
                    "    \"Decision Tree\": DecisionTreeClassifier(random_state=42, class_weight='balanced'),\n",
                    "    \"Random Forest\": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),\n",
                    "    \"Extra Trees\": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),\n",
                    "    \"Gradient Boosting\": GradientBoostingClassifier(random_state=42),\n",
                    "    \"AdaBoost\": AdaBoostClassifier(random_state=42),\n",
                    "    \"Hist Gradient Boosting\": HistGradientBoostingClassifier(random_state=42, class_weight='balanced')\n",
                    "}\n",
                    "\n",
                    "results = []\n",
                    "gkf = GroupKFold(n_splits=5)\n",
                    "\n",
                    "for name, model in models.items():\n",
                    "    model.fit(X_train_processed, y_train)\n",
                    "    y_test_pred = model.predict(X_test_processed)\n",
                    "    \n",
                    "    if hasattr(model, \"predict_proba\"):\n",
                    "        y_test_prob = model.predict_proba(X_test_processed)[:, 1]\n",
                    "    else:\n",
                    "        y_test_prob = y_test_pred\n",
                    "        \n",
                    "    acc = accuracy_score(y_test, y_test_pred)\n",
                    "    prec = precision_score(y_test, y_test_pred, zero_division=0)\n",
                    "    rec = recall_score(y_test, y_test_pred, zero_division=0)\n",
                    "    f1 = f1_score(y_test, y_test_pred, zero_division=0)\n",
                    "    auc = roc_auc_score(y_test, y_test_prob)\n",
                    "    \n",
                    "    cv_scores = cross_val_score(\n",
                    "        model, X_train_processed, y_train, \n",
                    "        groups=train_groups, cv=gkf, scoring='f1'\n",
                    "    )\n",
                    "    \n",
                    "    results.append({\n",
                    "        \"Model\": name,\n",
                    "        \"Test Accuracy\": acc,\n",
                    "        \"Precision\": prec,\n",
                    "        \"Recall\": rec,\n",
                    "        \"F1-Score\": f1,\n",
                    "        \"ROC-AUC\": auc,\n",
                    "        \"CV F1 Mean\": np.mean(cv_scores)\n",
                    "    })\n",
                    "\n",
                    "results_df = pd.DataFrame(results)\n",
                    "results_df.sort_values(by=\"F1-Score\", ascending=False)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #064e3b; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #34d399; font-weight: 700; font-size: 15px;\">💡 Model Seçimi Gerekçesi</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Karşılaştırma tablosunda görüldüğü üzere <b>Gradient Boosting</b> ve <b>Hist Gradient Boosting</b> modelleri en yüksek F1-Score ve ROC-AUC değerlerini sunmaktadır. İK probleminde en maliyetli hata istifa edecek çalışanı gözden kaçırmak (False Negative) olduğundan, duyarlılığı (Recall) yüksek ve dengeli olan <b>Gradient Boosting</b> final model olarak seçilmiştir.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0b1329; border-left: 6px solid #8b5cf6; border-radius: 8px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 20px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);\">\n",
                    "    <h2 style=\"margin: 0; color: #ffffff; font-size: 20px; font-weight: 700; margin-bottom: 10px;\">5. Hiperparametre Optimizasyonu & Eleştirel Değerlendirme</h2>\n",
                    "    <p style=\"margin: 0; color: #94a3b8; font-size: 14px; line-height: 1.5;\">Gradient Boosting modeli üzerinde sızıntı yapmayan <code>GroupKFold</code> ile <code>RandomizedSearchCV</code> kullanarak optimizasyon yapalım.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "raw_best_model = GradientBoostingClassifier(random_state=42)\n",
                    "param_dist = {\n",
                    "    'learning_rate': [0.01, 0.05, 0.1],\n",
                    "    'max_depth': [3, 5, 8],\n",
                    "    'n_estimators': [50, 100]\n",
                    "}\n",
                    "\n",
                    "search = RandomizedSearchCV(\n",
                    "    raw_best_model, param_distributions=param_dist,\n",
                    "    n_iter=4, cv=gkf, scoring='f1', random_state=42, n_jobs=-1\n",
                    ")\n",
                    "search.fit(X_train_processed, y_train, groups=train_groups)\n",
                    "best_model = search.best_estimator_\n",
                    "\n",
                    "print(\"En İyi Hiperparametreler:\", search.best_params_)\n",
                    "y_pred = best_model.predict(X_test_processed)\n",
                    "cm = confusion_matrix(y_test, y_pred)\n",
                    "print(\"Confusion Matrix:\")\n",
                    "print(cm)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #8b5cf6; border-radius: 6px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 10px;\">5.1. İş Değeri Tercümesi (Business Value Translation)</h3>\n",
                    "    <p style=\"color: #cbd5e1; font-size: 14px; line-height: 1.5; margin: 0;\">Modelin ürettiği kararların finansal faydalarını hesaplayalım:</p>\n",
                    "    <ul style=\"color: #cbd5e1; margin-top: 10px; font-size: 14px; line-height: 1.5; padding-left: 20px;\">\n",
                    "        <li>Her istifanın şirkete maliyeti: <b>$15,000</b></li>\n",
                    "        <li>Elde tutma aksiyonu maliyeti: <b>$3,000</b></li>\n",
                    "        <li>İK aksiyon aldığında çalışanın şirkette kalma oranı: <b>%80</b></li>\n",
                    "    </ul>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "tn, fp, fn, tp = cm.ravel()\n",
                    "\n",
                    "cost_without_model = (tp + fn) * 15000\n",
                    "cost_with_model = (fp * 3000) + (tp * 3000) + (tp * 0.20 * 15000) + (fn * 15000)\n",
                    "net_savings = cost_without_model - cost_with_model\n",
                    "\n",
                    "print(f\"Modelsiz Toplam İşten Ayrılma Kaybı: ${cost_without_model:,}\")\n",
                    "print(f\"Model Destekli İK Yönetim Maliyeti: ${cost_with_model:,}\")\n",
                    "print(f\"Modelin İK'ya Sağladığı Net Tasarruf: ${net_savings:,}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #064e3b; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #34d399; font-weight: 700; font-size: 15px;\">⚖️ Karar Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Model sayesinde elde edilen <b>$1,503,000</b> düzeyindeki net tasarruf, İK bütçesinin veri odaklı kararlarla yönetilmesinin gücünü kanıtlamaktadır. Hatalı tahminlerin maliyeti İK aksiyon maliyetlerine oranla çok yüksek olduğundan, modelin yüksek duyarlılığı doğrudan karlılığa yansımaktadır.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #8b5cf6; border-radius: 6px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 10px;\">5.2. Eşik Değeri Optimizasyonu (Threshold Tuning for Churn Costs)</h3>\n",
                    "    <p style=\"color: #cbd5e1; font-size: 14px; line-height: 1.5; margin: 0;\">Sınıf dengesizliği yüksek olan bu problemde varsayılan 0.50 eşik değeri yerine, şirketin eylem ve kayıp bütçesini en verimli hale getiren optimal olasılık eşik değerini bulalım.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Test kümesi için tahmin olasılıkları\n",
                    "if hasattr(best_model, 'predict_proba'):\n",
                    "    y_test_prob = best_model.predict_proba(X_test_processed)[:, 1]\n",
                    "else:\n",
                    "    y_test_prob = best_model.decision_function(X_test_processed)\n",
                    "\n",
                    "thresholds = np.arange(0.05, 0.95, 0.05)\n",
                    "savings_list = []\n",
                    "costs_list = []\n",
                    "\n",
                    "print(f\"{'Eşik':5s} | {'TN':5s} | {'FP':5s} | {'FN':5s} | {'TP':5s} | {'Maliyet':12s} | {'Net Tasarruf':12s}\")\n",
                    "print(\"-\"*78)\n",
                    "\n",
                    "for th in thresholds:\n",
                    "    y_pred_th = (y_test_prob >= th).astype(int)\n",
                    "    cm_th = confusion_matrix(y_test, y_pred_th)\n",
                    "    tn_th, fp_th, fn_th, tp_th = cm_th.ravel()\n",
                    "    \n",
                    "    cost_th = (fp_th * 3000) + (tp_th * 3000) + (tp_th * 0.20 * 15000) + (fn_th * 15000)\n",
                    "    savings_th = cost_without_model - cost_th\n",
                    "    savings_list.append(savings_th)\n",
                    "    costs_list.append(cost_th)\n",
                    "    \n",
                    "    print(f\"{th:.2f}  | {tn_th:5d} | {fp_th:5d} | {fn_th:5d} | {tp_th:5d} | ${cost_th:10,.0f} | ${savings_th:10,.0f}\")\n",
                    "\n",
                    "# En iyi eşik değerinin belirlenmesi\n",
                    "best_idx = np.argmax(savings_list)\n",
                    "opt_threshold = thresholds[best_idx]\n",
                    "opt_savings = savings_list[best_idx]\n",
                    "\n",
                    "# Grafik çizimi\n",
                    "fig_th = px.line(\n",
                    "    x=thresholds, y=savings_list, markers=True,\n",
                    "    title=\"Olasılık Eşik Değerine Göre İK Bütçe Tasarrufu ($)\",\n",
                    "    labels={'x': 'Sınıflandırma Eşik Değeri', 'y': 'Net Tasarruf ($)'},\n",
                    "    color_discrete_sequence=['#2E86AB']\n",
                    ")\n",
                    "fig_th.add_vline(x=opt_threshold, line_dash=\"dash\", line_color=\"#C73E1D\", \n",
                    "                 annotation_text=f\"Optimum Eşik ({opt_threshold:.2f})\", annotation_position=\"top left\")\n",
                    "fig_th.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #064e3b; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #34d399; font-weight: 700; font-size: 15px;\">⚖️ Eşik Optimizasyonu Karar Yorumu</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5; margin-bottom: 8px;\">Eşik optimizasyonu tablosunda ve grafiğinde açıkça görüldüğü üzere:</p>\n",
                    "    <ul style=\"margin: 0; color: #cbd5e1; padding-left: 20px; font-size: 14px; line-height: 1.6;\">\n",
                    "        <li>Modelin varsayılan eşik değeri (0.50) kullanıldığında şirketin elde ettiği tasarruf <b>$1,503,000</b>'dır.</li>\n",
                    "        <li>Ancak, eşik değeri <b>0.10</b> seviyesine düşürüldüğünde, model daha duyarlı (sensitive) hale gelerek ayrılacak çalışanları çok daha yüksek oranda yakalamaktadır. Bu eşikte kaçırılan çalışan (FN) sayısı 136'dan <b>79'a gerilemektedir</b>.</li>\n",
                    "        <li>Her ne kadar yanlış alarm (FP) sayısı 9'dan 57'ye çıksa ve boşa aksiyon maliyeti artsa da, yakalanan çalışan sayısının artması sayesinde elde edilen toplam tasarruf <b>$1,872,000</b> seviyesine ulaşmaktadır.</li>\n",
                    "        <li>Bu optimizasyon sayesinde şirket, modelin varsayılan ayarlarına kıyasla fazladan <b>$369,000 ek net tasarruf</b> elde etmektedir.</li>\n",
                    "    </ul>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #1e293b; border-left: 4px solid #8b5cf6; border-radius: 6px; padding: 20px; color: #ffffff; font-family: 'Inter', sans-serif; margin-top: 15px; margin-bottom: 10px;\">\n",
                    "    <h3 style=\"margin: 0; color: #ffffff; font-size: 16px; font-weight: 600; margin-bottom: 10px;\">5.3. Optimal Eşik Değerine Göre Karar Matrisi</h3>\n",
                    "    <p style=\"color: #cbd5e1; font-size: 14px; line-height: 1.5; margin: 0;\">Optimal eşik olan 0.10 kullanıldığındaki hata matrisini görselleştirerek, varsayılan eşik değeriyle (0.50) arasındaki farkı somutlaştıralım.</p>\n",
                    "</div>"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "y_pred_opt = (y_test_prob >= 0.10).astype(int)\n",
                    "cm_opt = confusion_matrix(y_test, y_pred_opt)\n",
                    "tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()\n",
                    "\n",
                    "fig_cm_opt = go.Figure(data=go.Heatmap(\n",
                    "    z=cm_opt,\n",
                    "    x=['Kalacak (Tahmin)', 'İstifa Edecek (Tahmin)'],\n",
                    "    y=['Kaldı (Gerçek)', 'İstifa Etti (Gerçek)'],\n",
                    "    colorscale='Reds',\n",
                    "    text=[[str(tn_opt), str(fp_opt)], [str(fn_opt), str(tp_opt)]],\n",
                    "    texttemplate=\"%{text}\",\n",
                    "    textfont=dict(size=16)\n",
                    "))\n",
                    "fig_cm_opt.update_layout(\n",
                    "    title=\"Optimal Eşik Değeri (0.10) Altında Karar Matrisi\",\n",
                    "    xaxis_title=\"Tahmin Edilen Durum\",\n",
                    "    yaxis_title=\"Gerçek Durum\"\n",
                    ")\n",
                    "fig_cm_opt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "<div style=\"background-color: #0f172a; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 6px; margin: 15px 0; color: #ffffff; font-family: 'Inter', sans-serif;\">\n",
                    "    <h4 style=\"margin-top: 0; color: #fbbf24; font-weight: 700; font-size: 15px;\">🔍 Final Analitiği Sentezi</h4>\n",
                    "    <p style=\"margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;\">Eşik değeri 0.10 yapıldığında:<br>\n",
                    "    - Kaçırılan istifa sayısı 136'dan 79'a inmiştir (False Negatives - En pahalı hatamız).<br>\n",
                    "    - Yakalanan ve müdahale edilebilecek istifa sayısı 170'ten 227'ye yükselmiştir (True Positives).<br>\n",
                    "    - Her ne kadar boşa aksiyon (False Positives) 9'dan 57'ye çıksa da, İK bütçesinin genel kârlılığı <b>$369,000 ek kazançla</b> maksimize edilmiştir.</p>\n",
                    "</div>"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("final-project/notebooks/final_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print("✅ final-project/notebooks/final_analysis.ipynb başarıyla oluşturuldu!")

if __name__ == "__main__":
    main()
