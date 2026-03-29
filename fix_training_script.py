"""
Fix remaining Turkish in train_all_models.py
"""

translations = {
    "✓ Logistic Regression başarıyla eğitildi": "✓ Logistic Regression trained successfully",
    "Logistic Regression eğitim errorsı:": "Logistic Regression training error:",
    "[2/2] Neural Network eğitiliyor...": "[2/2] Training Neural Network...",
    "✓ Neural Network başarıyla eğitildi": "✓ Neural Network trained successfully",
    "Neural Network eğitim errorsı:": "Neural Network training error:",
    "Tahminleme modellerini eğit ve karşılaştır.": "Train and compare forecasting models.",
    "Eğitilecek Modeller:": "Models to train:",
    "zaman serisi": "time series",
    "Model kayıt dizini": "Model save directory",
    "Tüm modellerle ilgili infoler": "Information about all models",
    "\"=\"*60": "\"=\"*60",
    "TAHMİNLEME MODELLERİ EĞİTİMİ BAŞLANIYOR": "FORECASTING MODELS TRAINING STARTING",
    "Test seti oranı": "Test set ratio",
    "zaman serisi": "time series",
    "[1/3] Linear Forecaster eğitiliyor...": "[1/3] Training Linear Forecaster...",
    "Tahmin yap": "Make predictions",
    "Model kaydet": "Save model",
    "✓ Linear Forecaster başarıyla eğitildi": "✓ Linear Forecaster trained successfully",
    "Linear Forecaster eğitim errorsı:": "Linear Forecaster training error:",
    "[2/3] ARIMA eğitiliyor...": "[2/3] Training ARIMA...",
    "✓ ARIMA başarıyla eğitildi": "✓ ARIMA trained successfully",
    "ARIMA eğitim errorsı:": "ARIMA training error:",
    "[3/3] Prophet eğitiliyor...": "[3/3] Training Prophet...",
    "✓ Prophet başarıyla eğitildi": "✓ Prophet trained successfully",
    "Prophet eğitim errorsı:": "Prophet training error:",
    "Training raporunu oluştur.": "Generate training report.",
    "Sınıflandırma sonuçları": "Classification results",
    "Tahminleme sonuçları": "Forecasting results",
    "Rapor dosya adı": "Report file name",
    "JSON olarak kaydet": "Save as JSON",
    "Metrics içindeki numpy arrays'i serializable yap": "Make numpy arrays in metrics serializable",
    "✓ Rapor kaydedildi:": "✓ Report saved:",
    "Konsola yazdır": "Print to console",
    "ÖZET:": "SUMMARY:",
    "Sınıflandırma Modelleri:": "Classification Models:",
    "Forecasting Models:": "Forecasting Models:",
    "Ana eğitim fonksiyonu": "Main training function",
    "[STEP 1] Sahte veri oluşturuluyor...": "[STEP 1] Creating fake data...",
    "Sınıflandırma verisi": "Classification data",
    "Kişi 1'in verisi geldiğinde düzenlenecek": "Will be updated when Person 1's data arrives",
    "Tahminleme verisi (zaman serisi)": "Forecasting data (time series)",
    "✓ Sahte veri oluşturuldu": "✓ Fake data created",
    "[STEP 2] Veri hazırlanıyor...": "[STEP 2] Preparing data...",
    "Dengeleme": "Balancing",
    "Train/Test split": "Train/Test split",
    "Normalization": "Normalization",
    "✓ Veri hazırlandı": "✓ Data prepared",
    "SINIFLAMA MODELLERİ": "CLASSIFICATION MODELS",
    "TAHMİNLEME MODELLERİ": "FORECASTING MODELS",
    "RAPOR OLUŞTUR": "GENERATE REPORT",
    "🎉 EĞITIM TAMAMLANDI": "🎉 TRAINING COMPLETED",
    "EĞITIM RAPORU OLUŞTURULUYOR": "GENERATING TRAINING REPORT",
    "Training pipeline errorsı:": "Training pipeline error:",
    "SINIFLAMA MODELLERİ EĞİTİMİ BAŞLANIYOR": "CLASSIFICATION MODELS TRAINING STARTING",
    "Sınıflandırma modellerini eğit ve karşılaştır.": "Train and compare classification models.",
}

with open(r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\train_all_models.py", 'r', encoding='utf-8') as f:
    content = f.read()

for turkish, english in sorted(translations.items(), key=lambda x: -len(x[0])):
    content = content.replace(turkish, english)

with open(r"c:\Users\Ahmet Hakan\OneDrive\Desktop\Data-Science-Project\train_all_models.py", 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ train_all_models.py fixed!")
