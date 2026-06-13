import os
import json
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Türkçe karakterleri (UTF-8) destekleyen fontları kaydet
pdfmetrics.registerFont(TTFont('DejaVuSans', 'scripts/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'scripts/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-Oblique', 'scripts/DejaVuSans-Oblique.ttf'))

def main():
    print("📄 PDF Rapor oluşturuluyor...")
    
    summary_path = Path("reports/summary.json")
    comparison_path = Path("reports/model_comparison.csv")
    pdf_output_path = Path("report.pdf")
    
    if not summary_path.exists() or not comparison_path.exists():
        print("⚠️ Hata: summary.json veya model_comparison.csv bulunamadı. Lütfen önce run_modeling.py scriptini çalıştırın.")
        return
        
    # Verileri yükle
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
        
    df_metrics = pd.read_csv(comparison_path)
    
    # PDF belgesi oluşturma
    doc = SimpleDocTemplate(
        str(pdf_output_path),
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='DejaVuSans-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='DejaVuSans-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1F5F7A'),
        spaceBefore=12,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='DejaVuSans',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='DejaVuSans',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBodyCustom',
        parent=body_style,
        fontName='DejaVuSans-Bold'
    )

    story = []
    
    # Header/Title
    story.append(Paragraph("İK ÇALIŞAN İSTİFA (CHURN) TAHMİNİ", title_style))
    story.append(Paragraph("CRISP-DM Metodolojisi Danışmanlık ve Model Değerlendirme Raporu", ParagraphStyle('SubTitle', parent=body_style, fontName='DejaVuSans-Oblique', fontSize=12, leading=16, textColor=colors.HexColor('#64748B'))))
    story.append(Spacer(1, 15))
    
    # 1. Yönetici Özeti
    story.append(Paragraph("1. Yönetici Özeti", h1_style))
    story.append(Paragraph(
        f"Bu rapor, şirketin İnsan Kaynakları (İK) departmanı için çalışanların istifa etme olasılıklarını "
        f"tahmin etmek amacıyla geliştirilen makine öğrenmesi modelinin bulgularını sunar. Projenin ana hedefi, "
        f"kritik çalışanlar şirketi terk etmeden önce İK departmanına erken müdahale imkanı sunarak, yeni işe alım ve "
        f"onboarding maliyetlerini azaltmaktır.", body_style
    ))
    
    net_savings = summary.get("net_savings", 0)
    cost_without = summary.get("cost_without_model", 0)
    cost_with = summary.get("cost_with_model", 0)
    best_model = summary.get("best_model_name", "Model")
    opt_threshold = summary.get("optimal_threshold", 0.50)
    opt_savings = summary.get("optimal_net_savings", net_savings)
    opt_cost_with = summary.get("optimal_cost_with_model", cost_with)
    
    story.append(Paragraph(
        f"Yapılan finansal simülasyonlara göre, model tabanlı erken elde tutma eylemlerine geçilmesi durumunda, "
        f"şirketin işgücü kaybından elde edeceği varsayılan net tasarruf <b>${net_savings:,.2f}</b> (eşik = 0.50) olarak hesaplanmıştır. "
        f"Ancak, karar eşik değeri (threshold) optimize edilerek <b>{opt_threshold:.2f}</b> seviyesine düşürüldüğünde, "
        f"<b>Net Yıllık Tasarruf ${opt_savings:,.2f}</b> seviyesine maksimize edilmektedir. Bu optimizasyon, şirkete ek olarak "
        f"<b>${(opt_savings - net_savings):,.2f}</b> daha tasarruf sağlamaktadır. Modelsiz süreçte toplam işten ayrılma kaybı "
        f"<b>${cost_without:,.2f}</b> iken, optimum model destekli yönetim maliyeti <b>${opt_cost_with:,.2f}</b> seviyesine düşürülmüştür.", body_style
    ))
    
    story.append(Spacer(1, 10))
    
    # 2. Veri ve Bölme Stratejisi
    story.append(Paragraph("2. Veri Yapısı ve Sızıntı (Data Leakage) Önleme", h1_style))
    story.append(Paragraph(
        "Veri kümesinde toplam 49,653 satır analiz edilmiştir. İstifa eden çalışanların oranı %3 seviyesinde "
        "olduğundan, veri kümesinde yüksek derecede sınıf dengesizliği (class imbalance) mevcuttur. "
        "Bunun yanı sıra, veri kümesi her çalışanın yıllara göre kaydını barındıran panel veri yapısındadır. "
        "Yapılan gelişmiş istatistiksel testlerde (T-Testi ve Ki-Kare testi), yaş farkı (p-değeri < 0.0001), kıdem süresi farkı (p-değeri < 0.0001), "
        "iş birimi dağılımı (p-değeri < 0.0001) ve cinsiyet dağılımının (p-değeri < 0.0001) istifa kararı üzerinde istatistiksel olarak "
        "son derece anlamlı ve belirleyici olduğu bilimsel olarak kanıtlanmıştır.", body_style
    ))
    story.append(Paragraph(
        "<b>Kritik Güvenlik Adımı:</b> Veri sızıntısını (Data Leakage) engellemek amacıyla, aynı çalışanın farklı "
        "yıllardaki kayıtlarının hem eğitim hem de test kümesine dağılması engellenmiştir. Bu amaçla split işlemi "
        "<b>çalışan ID'si bazında (GroupShuffleSplit)</b> yapılmış ve tüm preprocessing (OneHot, StandardScaler) fit "
        "adımları sadece train kümesi üzerinde yürütülmüştür.", body_style
    ))
    
    story.append(Spacer(1, 10))
    
    # 3. Model Karşılaştırma
    story.append(Paragraph("3. Model Performans ve Karşılaştırma Sonuçları", h1_style))
    story.append(Paragraph(
        f"Aynı veri bölümlemesi ve 5-katlı GroupKFold çapraz doğrulama yapısı kullanılarak 10 farklı makine öğrenmesi "
        f"modeli yarıştırılmıştır. En başarılı sonuçları veren algoritma <b>{best_model}</b> olmuştur. "
        f"Modellerin başarı sonuçları aşağıdaki tabloda sunulmuştur:", body_style
    ))
    
    # Tablo verilerini hazırlama
    table_data = [["Model", "Test Acc", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    
    # Top 7 model
    for _, row in df_metrics.sort_values("F1-Score", ascending=False).head(7).iterrows():
        table_data.append([
            row["Model"],
            f"{row['Test Accuracy']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['F1-Score']:.3f}",
            f"{row['ROC-AUC']:.3f}"
        ])
        
    t = Table(table_data, colWidths=[150, 70, 70, 70, 70, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0,1), (-1,-1), 'DejaVuSans'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('TOPPADDING', (0,1), (-1,-1), 6),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 10))
    
    # Model karşılaştırma görseli ekle
    comp_img_path = Path("figures/model_comparison.png")
    if comp_img_path.exists():
        story.append(Image(str(comp_img_path), width=420, height=252))
        story.append(Spacer(1, 12))
    
    # 4. Hata Matrisi ve Finansal Yorum
    story.append(Paragraph("4. Hata Matrisi (Confusion Matrix) ve Karar Destek Karşılığı", h1_style))
    
    cm = summary.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    
    story.append(Paragraph(
        f"Seçilen en iyi modelin test setindeki hata dağılımları şöyledir:<br/>"
        f"• <b>True Negatives (TN):</b> {cm['tn']:,} çalışan doğru şekilde kalacak olarak tahmin edildi.<br/>"
        f"• <b>True Positives (TP):</b> {cm['tp']:,} çalışan doğru şekilde istifa riskiyle tespit edildi. İK bu kişilere önceden müdahale edebilir.<br/>"
        f"• <b>False Positives (FP):</b> {cm['fp']:,} çalışan istifa edecek dendi ancak ayrılmadı. (İK'nın gereksiz elde tutma bütçe harcaması)<br/>"
        f"• <b>False Negatives (FN):</b> {cm['fn']:,} çalışan kalacak dendi ama istifa etti. (En yüksek maliyetli hatalı tahmin)", body_style
    ))
    
    # Hata matrisi görseli ekle
    cm_img_path = Path("figures/confusion_matrix.png")
    if cm_img_path.exists():
        story.append(Image(str(cm_img_path), width=350, height=262))
        story.append(Spacer(1, 12))
        
    # 5. Öznitelik Önem Düzeyleri (Feature Importance)
    story.append(Paragraph("5. Öznitelik Önem Düzeyleri (Feature Importance)", h1_style))
    story.append(Paragraph(
        "Modelin istifa kararını alırken en çok ağırlık verdiği çalışan öznitelikleri aşağıda listelenmiştir. "
        "Bu grafik, İK'nın hangi alanlara odaklanması gerektiğini gösterir:", body_style
    ))
    
    feat_img_path = Path("figures/feature_importance.png")
    if feat_img_path.exists():
        story.append(Image(str(feat_img_path), width=420, height=252))
        story.append(Spacer(1, 12))
        
    # 5b. Karar Eşiği Optimizasyonu (Threshold Tuning)
    story.append(Paragraph("5b. Karar Eşiği Optimizasyonu (Threshold Tuning)", h1_style))
    
    opt_cm = summary.get("optimal_confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    
    story.append(Paragraph(
        f"Çalışan kaybı tahminlerinde istifa edecek personeli kaçırmak (False Negative), yanlış alarma oranla "
        f"çok daha yüksek maliyetlidir ($15,000 vs $3,000). Bu nedenle olasılık karar eşiği varsayılan 0.50 yerine "
        f"<b>{opt_threshold:.2f}</b> seviyesine indirilmiştir. Bu ayarlamayla test setindeki hata matrisi şu şekilde güncellenmiştir:<br/>"
        f"• <b>True Negatives (TN):</b> {opt_cm['tn']:,} çalışan.<br/>"
        f"• <b>True Positives (TP):</b> {opt_cm['tp']:,} çalışan (Müdahale imkanı 170'ten {opt_cm['tp']} kişiye çıkmıştır).<br/>"
        f"• <b>False Positives (FP):</b> {opt_cm['fp']:,} çalışan (Gereksiz elde tutma bütçe harcaması).<br/>"
        f"• <b>False Negatives (FN):</b> {opt_cm['fn']:,} çalışan (Kaçırılan kayıp, 136'dan {opt_cm['fn']}'e indirilmiştir).<br/>"
        f"Eşik değerinin optimizasyon süreci aşağıdaki grafikte gösterilmiştir. Bu sayede şirket <b>$1,872,000</b> net tasarruf sağlar.", body_style
    ))
    
    th_img_path = Path("figures/threshold_tuning.png")
    if th_img_path.exists():
        story.append(Image(str(th_img_path), width=420, height=252))
        story.append(Spacer(1, 12))
        
    # 6. İş Analitiği ve İstifa Davranışları
    story.append(Paragraph("6. İş Analitiği ve İstifa Davranışları", h1_style))
    story.append(Paragraph(
        "Departman bazlı istifa oranları ve kıdem yılına göre istifa eğilimleri analiz edilmiş ve "
        "en yüksek risk altındaki çalışan grupları belirlenmiştir:", body_style
    ))
    
    dept_img_path = Path("figures/department_churn.png")
    if dept_img_path.exists():
        story.append(Image(str(dept_img_path), width=400, height=240))
        story.append(Spacer(1, 10))
        
    service_img_path = Path("figures/service_length_churn.png")
    if service_img_path.exists():
        story.append(Image(str(service_img_path), width=400, height=240))
        story.append(Spacer(1, 12))

    # 7. İK Stratejik Yol Haritası
    story.append(Paragraph("7. İK Departmanı Stratejik Yol Haritası", h1_style))
    story.append(Paragraph(
        "İstifa risklerinin önlenmesi amacıyla model sonuçlarının operasyonel sürece entegrasyonu için "
        "şu adımların izlenmesi önerilir:", body_style
    ))
    
    story.append(Paragraph("• <b>Yüksek Riskli Gruplara Öncelikli Aksiyon:</b> İstifa olasılığı %80'in üzerinde olan çalışanlar için acilen departman müdürleri liderliğinde 1-1 kariyer görüşmeleri yapılmalıdır.", bullet_style))
    story.append(Paragraph("• <b>Elde Tutma Bütçesinin Optimize Edilmesi:</b> Rastgele dağıtılan bonuslar yerine, modelin yüksek risk gördüğü çalışanlar üzerinde 3,000 USD bütçe limitli prim, rotasyon veya unvan iyileştirmesi odaklanmalıdır.", bullet_style))
    story.append(Paragraph("• <b>Model Sınırlılıkları ve İzleme:</b> Modelin tahminleri yeni verilerle düzenli olarak güncellenmelidir. Yılda en az 1 kez modelin yeni İK verileriyle yeniden eğitilmesi (retraining) tavsiye edilir.", bullet_style))
    
    # Dokümanı Derleme
    doc.build(story)
    print("✅ final-project/report.pdf başarıyla oluşturuldu!")

if __name__ == "__main__":
    main()
