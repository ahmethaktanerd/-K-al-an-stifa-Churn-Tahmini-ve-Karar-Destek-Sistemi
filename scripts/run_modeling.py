import os
import time
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Sklearn imports
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

# Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.ensemble import HistGradientBoostingClassifier
from prettytable import PrettyTable

def main():
    print("🚀 CRISP-DM Machine Learning Pipeline başlatılıyor...")

    # Klasörlerin varlığını garantile
    Path('final-project/data/processed').mkdir(parents=True, exist_ok=True)
    Path('final-project/models').mkdir(parents=True, exist_ok=True)
    Path('final-project/figures').mkdir(parents=True, exist_ok=True)
    Path('final-project/reports').mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. DATA UNDERSTANDING & EDA
    # ----------------------------------------------------
    print("\n🔍 AŞAMA 1: Data Understanding & EDA...")
    df = pd.read_csv('final-project/data/raw/veri_seti.csv')
    print(f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # Hedef Değişken Dağılımı
    status_counts = df['STATUS'].value_counts()
    status_ratio = df['STATUS'].value_counts(normalize=True) * 100
    print(f"Sınıf Dağılımı:\nACTIVE: {status_counts.get('ACTIVE', 0)} (%{status_ratio.get('ACTIVE', 0):.2f})\nTERMINATED: {status_counts.get('TERMINATED', 0)} (%{status_ratio.get('TERMINATED', 0):.2f})")
    
    # Eksik değer kontrolü
    missing_data = df.isnull().sum()
    print("Eksik veri adedi:", missing_data.sum())
    
    # ----------------------------------------------------
    # 2. DATA PREPARATION (Leakage-Free Preprocessing)
    # ----------------------------------------------------
    print("\n🧹 AŞAMA 2: Data Preparation & Preprocessing...")
    
    # Hedef değişkeni ikili (binary) formata çevirme (TERMINATED = 1, ACTIVE = 0)
    df['target'] = (df['STATUS'] == 'TERMINATED').astype(int)
    
    # Hedef sızıntısı (leakage) yaratan sütunları ve tanımlayıcıları ayırma
    # terminationdate_key, termreason_desc, termtype_desc doğrudan hedef sızıntısıdır.
    # EmployeeID ise veri kümesinde çoklama olduğu için modelin ezberlemesini önlemek adına gruplama için ayrılacaktır.
    leakage_cols = ['terminationdate_key', 'termreason_desc', 'termtype_desc', 'STATUS', 'gender_short']
    id_cols = ['EmployeeID', 'recorddate_key', 'birthdate_key', 'orighiredate_key']
    
    X = df.drop(columns=['target'] + leakage_cols + id_cols)
    y = df['target']
    groups = df['EmployeeID']
    
    print("Kullanılacak Öznitelikler (Features):", list(X.columns))
    
    # Kategorik ve sayısal sütunların belirlenmesi
    categorical_cols = ['city_name', 'department_name', 'job_title', 'BUSINESS_UNIT', 'store_name']
    numerical_cols = ['age', 'length_of_service', 'STATUS_YEAR']
    
    # Model sızıntısını önlemek için en kritik adım: Train-Test Split (Grup Tabanlı)
    # Bir çalışanın bazı yılları eğitimde, bazı yılları testte olursa model sızıntısı olur.
    # Bu yüzden GroupShuffleSplit kullanarak çalışanları bütünsel olarak bölüyoruz.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    
    print(f"Eğitim Seti Boyutu: {X_train.shape[0]} (Benzersiz Çalışan: {train_groups.nunique()})")
    print(f"Test Seti Boyutu: {X_test.shape[0]} (Benzersiz Çalışan: {groups.iloc[test_idx].nunique()})")
    
    # Preprocessing Pipeline (Transformer)
    # Kategorik sütunları One-Hot, sayısal sütunları StandardScaler ile ölçekliyoruz
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Pipeline nesnesini oluştur ve train üzerinde fit et
    # Test setine fit uygulanmaz (Data Leakage engellenir)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, 'final-project/models/pipeline.joblib')
    print("Saved preprocessing pipeline to final-project/models/pipeline.joblib")
    
    # Verileri dönüştür
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Temizlenmiş veriyi kaydet (EDA çıktısı olarak processed dizinine)
    df_cleaned = df.drop(columns=leakage_cols)
    df_cleaned.to_csv('final-project/data/processed/veri_seti_cleaned.csv', index=False)
    print("Temizlenmiş veri seti kaydedildi: final-project/data/processed/veri_seti_cleaned.csv")

    # ----------------------------------------------------
    # 3. MODELING & COMPARISON (10 Classifiers)
    # ----------------------------------------------------
    print("\n🤖 AŞAMA 3: Modelleme ve Model Karşılaştırma...")
    
    models = {
        "Dummy Classifier": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Ridge Classifier": RidgeClassifier(random_state=42, class_weight='balanced'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    }
    
    results = []
    
    # GroupKFold CV Stratejisi (Leakage engellemek için CV de çalışan gruplarına göre yapılmalıdır)
    gkf = GroupKFold(n_splits=5)
    
    for name, model in models.items():
        print(f"Modelleniyor: {name}...")
        start_time = time.time()
        
        # Eğit
        model.fit(X_train_processed, y_train)
        train_time = time.time() - start_time
        
        # Tahminler
        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)
        
        # Olasılıklar (ROC-AUC için)
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train_processed)[:, 1]
            y_test_prob = model.predict_proba(X_test_processed)[:, 1]
        elif hasattr(model, "decision_function"):
            y_train_prob = model.decision_function(X_train_processed)
            y_test_prob = model.decision_function(X_test_processed)
        else:
            y_train_prob = y_train_pred
            y_test_prob = y_test_pred
            
        # Performans Metrikleri
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_prob)
        
        # Cross Validation (Eğitim setinde GroupKFold ile)
        # Ridge ve predict_proba desteklemeyenler için alternatif veya standard metrik
        cv_scores = cross_val_score(
            model, X_train_processed, y_train, 
            groups=train_groups, cv=gkf, scoring='f1'
        )
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        overfit_gap = train_acc - test_acc
        
        results.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Precision": round(test_prec, 4),
            "Recall": round(test_rec, 4),
            "F1-Score": round(test_f1, 4),
            "ROC-AUC": round(test_auc, 4),
            "CV F1 Mean": round(cv_mean, 4),
            "CV F1 Std": round(cv_std, 4),
            "Overfit Gap": round(overfit_gap, 4),
            "Train Time (s)": round(train_time, 3)
        })
        
    # Tablo ve CSV olarak kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv('final-project/reports/model_comparison.csv', index=False)
    
    # PrettyTable ile göster
    pt = PrettyTable()
    pt.field_names = ["Model", "Test Acc", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV F1", "Süre (s)"]
    for r in results:
        pt.add_row([
            r["Model"], r["Test Accuracy"], r["Precision"], r["Recall"], 
            r["F1-Score"], r["ROC-AUC"], f"{r['CV F1 Mean']} ± {r['CV F1 Std']}", r["Train Time (s)"]
        ])
    print("\n📊 Model Karşılaştırma Sonuçları:")
    print(pt)
    
    # ----------------------------------------------------
    # 4. EVALUATION & BEST MODEL SELECTION
    # ----------------------------------------------------
    print("\n⚖️ AŞAMA 4: Eleştirel Değerlendirme & En İyi Model Seçimi...")
    
    # En iyi modeli seç (F1 ve Recall dengesine göre - HR maliyet analizinde Recall çok önemlidir çünkü ayrılan çalışanları kaçırmamak gerekir)
    # Burada Hist Gradient Boosting ve Random Forest güçlü adaylar. F1-Score'u en yüksek olanı seçiyoruz.
    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    print(f"Seçilen En İyi Algoritma: {best_model_name}")
    
    # Hiperparametre Optimizasyonu
    print(f"🔧 {best_model_name} için Hiperparametre Optimizasyonu başlatılıyor...")
    raw_best_model = models[best_model_name]
    
    # En iyi modele göre arama uzayı tanımlama
    if best_model_name in ["Random Forest", "Extra Trees"]:
        param_dist = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    elif best_model_name in ["Gradient Boosting", "Hist Gradient Boosting"]:
        param_dist = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8]
        }
        if best_model_name == "Gradient Boosting":
            param_dist['n_estimators'] = [50, 100]
    else:
        if hasattr(raw_best_model, "C"):
            param_dist = {'C': [0.1, 1.0, 10.0]}
        elif hasattr(raw_best_model, "n_neighbors"):
            param_dist = {'n_neighbors': [3, 5, 7]}
        else:
            param_dist = {}
            
    if param_dist:
        # Veri sızıntısını önlemek için GroupKFold CV yapısı kullanıyoruz
        search = RandomizedSearchCV(
            raw_best_model, param_distributions=param_dist, 
            n_iter=4, cv=gkf, scoring='f1', random_state=42, n_jobs=-1
        )
        search.fit(X_train_processed, y_train, groups=train_groups)
        best_model = search.best_estimator_
        print("En İyi Hiperparametreler:", search.best_params_)
    else:
        best_model = raw_best_model
        best_model.fit(X_train_processed, y_train)
        
    joblib.dump(best_model, 'final-project/models/best_model.joblib')
    print("Saved best model to final-project/models/best_model.joblib")
    
    # En İyi Model Tahmin Değerlendirmesi
    y_test_pred = best_model.predict(X_test_processed)
    if hasattr(best_model, "predict_proba"):
        y_test_prob = best_model.predict_proba(X_test_processed)[:, 1]
    else:
        y_test_prob = best_model.decision_function(X_test_processed)
        
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"True Negative (Kalacak dedi, Kaldı): {tn}")
    print(f"False Positive (İstifa dedi, Kaldı - Boşa Aksiyon): {fp}")
    print(f"False Negative (Kalacak dedi, İstifa Etti - Kayıp Maliyet): {fn}")
    print(f"True Positive (İstifa dedi, İstifa Etti - Başarılı Tespit): {tp}")
    
    # İş Değeri Maliyet Analizi (Business Value Translation)
    # Ortalama bir çalışanın işten ayrılma maliyeti = 15,000 USD (ikame maliyeti)
    # Elde tutma aksiyonu (bonus/iyileştirme) maliyeti = 3,000 USD
    # İK aksiyon aldığında çalışanın kalma ihtimali = %80
    
    cost_without_model = (tp + fn) * 15000 # Her ayrılan çalışan 15k zarar
    
    # Model varken:
    # - FP olanlara boşuna 3k harcandı (harcama = fp * 3000)
    # - TP olanlara 3k harcandı. %80'i kaldı (maliyet = tp * 3000), %20'si yinede gitti (kayıp = tp * 0.20 * 15000)
    # - FN olanlar kaçırıldı, doğrudan gittiler (kayıp = fn * 15000)
    cost_with_model = (fp * 3000) + (tp * 3000) + (tp * 0.20 * 15000) + (fn * 15000)
    net_savings = cost_without_model - cost_with_model
    
    print(f"\n💼 Maliyet/İçgörü Analizi (Eşik = 0.50):")
    print(f"Modelsiz Toplam İşten Ayrılma Kaybı: ${cost_without_model:,}")
    print(f"Model Destekli İK Yönetim Maliyeti: ${cost_with_model:,}")
    print(f"Modelin İK'ya Sağladığı Net Tasarruf: ${net_savings:,}")

    # ----------------------------------------------------
    # Eşik Değeri Optimizasyonu (Threshold Tuning for Business Value)
    # ----------------------------------------------------
    print("\n🎯 Eşik Değeri Optimizasyonu (Threshold Tuning) hesaplanıyor...")
    best_threshold = 0.50
    max_savings = net_savings
    best_cost_with_model = cost_with_model
    best_cm_details = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    
    threshold_results = []
    for th in np.arange(0.05, 0.95, 0.05):
        y_pred_th = (y_test_prob >= th).astype(int)
        cm_th = confusion_matrix(y_test, y_pred_th)
        tn_th, fp_th, fn_th, tp_th = cm_th.ravel()
        
        cost_with_model_th = (fp_th * 3000) + (tp_th * 3000) + (tp_th * 0.20 * 15000) + (fn_th * 15000)
        savings_th = cost_without_model - cost_with_model_th
        
        threshold_results.append({
            "threshold": float(th),
            "tn": int(tn_th), "fp": int(fp_th), "fn": int(fn_th), "tp": int(tp_th),
            "cost_with_model": float(cost_with_model_th),
            "savings": float(savings_th)
        })
        
        if savings_th > max_savings:
            max_savings = savings_th
            best_threshold = float(th)
            best_cost_with_model = float(cost_with_model_th)
            best_cm_details = {"tn": int(tn_th), "fp": int(fp_th), "fn": int(fn_th), "tp": int(tp_th)}
            
    print(f"En İyi Eşik Değeri (Optimal Threshold): {best_threshold:.2f}")
    print(f"Maksimum Net Tasarruf: ${max_savings:,.2f} (Eşik 0.50 iken tasarruf: ${net_savings:,.2f})")
    
    # Static threshold tuning plot using matplotlib
    plt.figure(figsize=(10, 6))
    th_vals = [r["threshold"] for r in threshold_results]
    sav_vals = [r["savings"] for r in threshold_results]
    plt.plot(th_vals, sav_vals, marker='o', color='#2E86AB', lw=2, label='Tasarruf ($)')
    plt.axvline(x=best_threshold, linestyle='--', color='#C73E1D', lw=2, label=f"Optimum Eşik ({best_threshold:.2f})")
    plt.title("Olasılık Eşik Değerine Göre İK Bütçe Tasarrufu ($)", fontsize=14, fontweight='bold')
    plt.xlabel("Sınıflandırma Eşik Değeri")
    plt.ylabel("Net Tasarruf ($)")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("final-project/figures/threshold_tuning.png", dpi=300)
    plt.close()
    print("Saved threshold tuning plot to final-project/figures/threshold_tuning.png")

    
    # ----------------------------------------------------
    # 5. FIGURES & VISUALIZATIONS GENERATION
    # ----------------------------------------------------
    print("\n📈 AŞAMA 5: Grafikler oluşturuluyor...")
    
    # 1. Model Karşılaştırma Grafiği
    fig_comp = px.bar(
        results_df, x="Model", y="F1-Score", color="Recall",
        title="10 Farklı Modelin F1-Score ve Recall Karşılaştırması",
        labels={"F1-Score": "F1 Başarı Skoru", "Recall": "Recall (Duyarlılık)"},
        text="F1-Score", color_continuous_scale="Viridis"
    )
    fig_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_comp.write_html("final-project/figures/model_comparison.html")
    
    # Matplotlib static PNG
    plt.figure(figsize=(10, 6))
    sorted_df = results_df.sort_values(by="F1-Score", ascending=False)
    sns.barplot(data=sorted_df, x="Model", y="F1-Score", palette="viridis")
    plt.title("10 Farklı Modelin F1-Score Karşılaştırması", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("F1 Başarı Skoru")
    plt.tight_layout()
    plt.savefig("final-project/figures/model_comparison.png", dpi=300)
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Kalacak (Tahmin)', 'İstifa Edecek (Tahmin)'],
        y=['Kaldı (Gerçek)', 'İstifa Etti (Gerçek)'],
        colorscale='Blues',
        text=[[str(tn), str(fp)], [str(fn), str(tp)]],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    fig_cm.update_layout(title="Best Model Confusion Matrix", xaxis_title="Tahmin", yaxis_title="Gerçek")
    fig_cm.write_html("final-project/figures/confusion_matrix.html")
    
    # Matplotlib static PNG
    plt.figure(figsize=(8, 6))
    labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=['Kalacak (Tahmin)', 'İstifa Edecek (Tahmin)'],
                yticklabels=['Kaldı (Gerçek)', 'İstifa Etti (Gerçek)'],
                annot_kws={"size": 14})
    plt.title("En İyi Model Karmaşıklık Matrisi (Confusion Matrix)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("final-project/figures/confusion_matrix.png", dpi=300)
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    fig_roc = px.line(
        x=fpr, y=tpr,
        title=f"ROC Eğrisi (AUC = {results_df.loc[best_idx, 'ROC-AUC']:.4f})",
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
    )
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig_roc.write_html("final-project/figures/roc_curve.html")
    
    # Matplotlib static PNG
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86AB', lw=2, label=f"ROC Curve (AUC = {results_df.loc[best_idx, 'ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], color='#64748B', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("final-project/figures/roc_curve.png", dpi=300)
    plt.close()
    
    # 4. Correlation Matrix (Feature'lar için)
    corr_df = X_train[numerical_cols].copy()
    corr_df['Target'] = y_train
    corr_matrix = corr_df.corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".3f", 
        title="Sayısal Değişkenler ve Target Korelasyon Matrisi",
        color_continuous_scale="RdBu_r"
    )
    fig_corr.write_html("final-project/figures/correlation_matrix.html")
    
    # Matplotlib static PNG
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0)
    plt.title("Sayısal Değişkenler ve Target Korelasyon Matrisi", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("final-project/figures/correlation_matrix.png", dpi=300)
    plt.close()

    # 5. Feature Importance
    ohe = preprocessor.named_transformers_['cat']
    cat_features = list(ohe.get_feature_names_out(categorical_cols))
    all_features = numerical_cols + cat_features
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    fig_imp = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        title="En Önemli 10 Çalışan Özniteliği (Feature Importance)",
        color="Importance", color_continuous_scale="Viridis",
        labels={"Importance": "Önem Düzeyi", "Feature": "Öznitelik"}
    )
    fig_imp.write_html("final-project/figures/feature_importance.html")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis")
    plt.title("En Önemli 10 Çalışan Özniteliği (Feature Importance)", fontsize=14, fontweight='bold')
    plt.xlabel("Önem Düzeyi")
    plt.ylabel("Öznitelik")
    plt.tight_layout()
    plt.savefig("final-project/figures/feature_importance.png", dpi=300)
    plt.close()
    
    # 6. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve
    precisions_vals, recalls_vals, _ = precision_recall_curve(y_test, y_test_prob)
    fig_pr = px.line(
        x=recalls_vals, y=precisions_vals,
        title="Hassasiyet-Duyarlılık (Precision-Recall) Eğrisi",
        labels={'x': 'Duyarlılık (Recall)', 'y': 'Kesinlik (Precision)'}
    )
    fig_pr.write_html("final-project/figures/precision_recall_curve.html")
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls_vals, precisions_vals, color='#A23B72', lw=2, label="PR Curve")
    plt.xlabel('Duyarlılık (Recall)')
    plt.ylabel('Kesinlik (Precision)')
    plt.title('Hassasiyet-Duyarlılık (Precision-Recall) Eğrisi', fontsize=14, fontweight='bold')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig("final-project/figures/precision_recall_curve.png", dpi=300)
    plt.close()

    # 7. Bivariate Scatter Plot
    fig_biv = px.scatter(
        df, x="age", y="length_of_service", color="STATUS",
        color_discrete_sequence=['#2E86AB', '#C73E1D'],
        title="Yaş vs Hizmet Süresi ve Ayrılma Durumu Etkileşimi",
        labels={"age": "Yaş", "length_of_service": "Hizmet Süresi (Yıl)"}
    )
    fig_biv.write_html("final-project/figures/bivariate_scatter.html")
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="age", y="length_of_service", hue="STATUS", palette=['#2E86AB', '#C73E1D'], alpha=0.6)
    plt.title("Yaş vs Hizmet Süresi ve Ayrılma Durumu", fontsize=14, fontweight='bold')
    plt.xlabel("Yaş")
    plt.ylabel("Hizmet Süresi")
    plt.tight_layout()
    plt.savefig("final-project/figures/bivariate_scatter.png", dpi=300)
    plt.close()

    # 8. Departman Bazlı İstifa Oranları (Churn Rate by Department)
    dept_churn = df.groupby('department_name')['target'].mean().reset_index()
    dept_churn['target'] = dept_churn['target'] * 100
    dept_churn = dept_churn.sort_values('target', ascending=False)
    
    fig_dept_churn = px.bar(
        dept_churn, x="target", y="department_name", orientation="h",
        title="Departman Bazlı İstifa Oranları (%)",
        color="target", color_continuous_scale="Reds",
        labels={"target": "İstifa Oranı (%)", "department_name": "Departman"}
    )
    fig_dept_churn.write_html("final-project/figures/department_churn.html")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=dept_churn, x="target", y="department_name", palette="Reds_r")
    plt.title("Departman Bazlı İstifa Oranları (%)", fontsize=14, fontweight='bold')
    plt.xlabel("İstifa Oranı (%)")
    plt.ylabel("Departman")
    plt.tight_layout()
    plt.savefig("final-project/figures/department_churn.png", dpi=300)
    plt.close()
    
    # 9. Kıdem Yılına (Service Length) Göre İstifa Oranları (Tenure-based Churn Rate)
    service_churn = df.groupby('length_of_service')['target'].mean().reset_index()
    service_churn['target'] = service_churn['target'] * 100
    
    fig_service_churn = px.line(
        service_churn, x="length_of_service", y="target", markers=True,
        title="Kıdem Yılına Göre İstifa Oranları (%)",
        labels={"target": "İstifa Oranı (%)", "length_of_service": "Hizmet Süresi (Yıl)"}
    )
    fig_service_churn.write_html("final-project/figures/service_length_churn.html")
    
    plt.figure(figsize=(10, 6))
    plt.plot(service_churn["length_of_service"], service_churn["target"], marker='o', color='#C73E1D', lw=2)
    plt.title("Kıdem Yılına Göre İstifa Oranları (%)", fontsize=14, fontweight='bold')
    plt.xlabel("Hizmet Süresi (Yıl)")
    plt.ylabel("İstifa Oranı (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("final-project/figures/service_length_churn.png", dpi=300)
    plt.close()

    # Rapor ve PDF üretici için özet çıktı kaydı
    summary_data = {
        "best_model_name": best_model_name,
        "net_savings": float(net_savings),
        "cost_without_model": float(cost_without_model),
        "cost_with_model": float(cost_with_model),
        "metrics": results_df.to_dict(orient="records"),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "optimal_threshold": float(best_threshold),
        "optimal_net_savings": float(max_savings),
        "optimal_cost_with_model": float(best_cost_with_model),
        "optimal_confusion_matrix": {
            "tn": int(best_cm_details["tn"]),
            "fp": int(best_cm_details["fp"]),
            "fn": int(best_cm_details["fn"]),
            "tp": int(best_cm_details["tp"])
        }
    }
    with open('final-project/reports/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=4)
        
    print("\n✅ Modelleme ve Değerlendirme süreci başarıyla tamamlandı!")
    print("Modeller ve görseller 'final-project/models/' ve 'final-project/figures/' klasörlerine kaydedildi.")

if __name__ == "__main__":
    main()
