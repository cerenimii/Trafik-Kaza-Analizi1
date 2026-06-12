# Trafik-Kaza-Analizi1
# Trafik Kaza Tahmin Sistemi 🚗💥

Bu proje, Birleşik Krallık'ta 2005-2014 yılları arasında meydana gelen trafik kazalarını analiz ederek, bir kazanın ciddiyetini (ölümcül/ciddi veya hafif) tahmin eden bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır.

## Proje Hakkında

Trafik kazaları, dünya çapında önemli bir halk sağlığı sorunudur. Bu proje, kaza verilerini analiz ederek hangi faktörlerin kazanın ciddiyetini etkilediğini belirlemekte ve gelecekteki kazaların şiddetini tahmin etmeye çalışmaktadır.

##  Amaçlar

*   Kaza verilerini keşfederek anlamlı desenler ve ilişkiler bulmak.
*   Veri ön işleme adımları (eksik değer doldurma, aykırı değer temizliği, özellik mühendisliği) uygulamak.
*   **Random Forest, XGBoost ve Lojistik Regresyon** gibi sınıflandırma algoritmalarını eğiterek en başarılı modeli belirlemek.
*   **SHAP** değerleri ile modelin tahminlerini yorumlamak ve en önemli özellikleri tespit etmek.
*   **K-Means** kümeleme algoritması ile benzer özelliklere sahip kaza kümelerini oluşturmak.

##Kullanılan Teknolojiler ve Kütüphaneler

*   **Python 3.8+**
*   **Pandas & NumPy:** Veri manipülasyonu ve analizi için.
*   **Matplotlib & Seaborn:** Veri görselleştirme için.
*   **Scikit-learn:** Model eğitimi, veri ön işleme ve değerlendirme için.
*   **Imbalanced-learn (SMOTE):** Dengesiz veri setini dengelemek için.
*   **XGBoost:** Gradient boosting tabanlı sınıflandırma modeli için.
*   **SHAP:** Model yorumlanabilirliği için.

