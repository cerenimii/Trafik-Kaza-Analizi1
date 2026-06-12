# ============================================================
# TRAFFIC ACCIDENT SEVERITY PREDICTION SYSTEM
# 03 - Classification: Random Forest & XGBoost + SHAP
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, f1_score
)
import shap

print("=" * 60)
print("CLASSIFICATION MODELLERİ")
print("=" * 60)

# ============================================================
# 1. VERİYİ YÜKLE
# ============================================================
X_train, X_test, y_train, y_test, feature_names = joblib.load('preprocessed_data.pkl')
print(f"Eğitim: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 2. MODELLERİ TANIMLA
# ============================================================
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=sum(y_train==0)/sum(y_train==1),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
}

# ============================================================
# 3. MODELLERİ EĞİT VE DEĞERLENDİR
# ============================================================
results = {}

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"{'='*40}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = {
        'model': model,
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'cm': cm
    }

    print(f"Accuracy  : %{acc*100:.2f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"\nSınıflandırma Raporu:\n{classification_report(y_test, y_pred, target_names=['Hafif', 'Ciddi'])}")

# ============================================================
# 4. KARŞILAŞTIRMA TABLOSU
# ============================================================
print("\n" + "=" * 60)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 60)
compare_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy (%)': [f"%{r['accuracy']*100:.2f}" for r in results.values()],
    'F1-Score': [f"{r['f1']:.4f}" for r in results.values()],
    'ROC-AUC': [f"{r['auc']:.4f}" for r in results.values()],
}).set_index('Model')
print(compare_df)

# ============================================================
# 5. KARMASıKLIK MATRİSLERİ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Hafif', 'Ciddi'],
                yticklabels=['Hafif', 'Ciddi'])
    ax.set_title(f'{name}\nAccuracy: %{res["accuracy"]*100:.1f}', fontweight='bold')
    ax.set_ylabel('Gerçek')
    ax.set_xlabel('Tahmin')
plt.tight_layout()
plt.savefig('05_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Karmaşıklık matrisleri kaydedildi")

# ============================================================
# 6. ROC EĞRİSİ
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#1565C0', '#C62828', '#2E7D32']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{name} (AUC = {res["auc"]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Rastgele Sınıflandırıcı')
ax.set_xlabel('1 - Özgüllük (FPR)', fontsize=13)
ax.set_ylabel('Duyarlılık (TPR)', fontsize=13)
ax.set_title('ROC Eğrisi Karşılaştırması', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('06_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ ROC eğrisi kaydedildi")

# ============================================================
# 7. SHAP ANALİZİ - EN İYİ MODEL (XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("SHAP ANALİZİ - XGBoost")
print("=" * 60)

best_model = results['XGBoost']['model']

# SHAP değerlerini hesapla
explainer = shap.TreeExplainer(best_model)
X_test_df = pd.DataFrame(X_test, columns=feature_names)
shap_values = explainer.shap_values(X_test_df.sample(min(2000, len(X_test_df)), random_state=42))
X_sample = X_test_df.sample(min(2000, len(X_test_df)), random_state=42)

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                  max_display=15, plot_size=(12, 8))
plt.title("SHAP Feature Importance - XGBoost", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('07_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ SHAP bar plot kaydedildi")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15, plot_size=(12, 8))
plt.title("SHAP Değerleri - XGBoost", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_shap_dot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ SHAP dot plot kaydedildi")

# ============================================================
# 8. EN İYİ MODELİ KAYDET
# ============================================================
best_name = max(results, key=lambda x: results[x]['auc'])
joblib.dump(results[best_name]['model'], 'best_model.pkl')
print(f"\n✅ En iyi model: {best_name} (AUC={results[best_name]['auc']:.4f})")
print("✓ Model kaydedildi: best_model.pkl")
