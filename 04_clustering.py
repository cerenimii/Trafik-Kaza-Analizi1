# ============================================================
# TRAFFIC ACCIDENT SEVERITY PREDICTION SYSTEM
# 04 - K-Means Clustering + PCA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("K-MEANS CLUSTERING ANALİZİ")
print("=" * 60)

# ============================================================
# 1. VERİYİ YÜKLEME
# ============================================================
X_train, X_test, y_train, y_test, feature_names = joblib.load('preprocessed_data.pkl')

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# Örnekle (büyük veri için)
np.random.seed(42)
idx = np.random.choice(len(X_all), min(50000, len(X_all)), replace=False)
X_sample = X_all[idx]
y_sample = y_all[idx]

print(f"Kümeleme için kullanılan örnek: {X_sample.shape[0]:,} kayıt")

# ============================================================
# 2. STANDARTLAŞTIRMA
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# ============================================================
# 3. ELBOW YÖNTEMİ - Optimal K Belirleme
# ============================================================
print("\nElbow yöntemi ile optimal k belirleniyor...")
inertias = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k}: Inertia = {km.inertia_:.0f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Küme Sayısı (k)', fontsize=13)
ax.set_ylabel('Inertia', fontsize=13)
ax.set_title('Elbow Yöntemi - Optimal Küme Sayısı', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('09_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Elbow grafiği kaydedildi")

# ============================================================
# 4. K-MEANS UYGULAMA (k=4)
# ============================================================
optimal_k = 4
print(f"\nK-Means uygulanıyor: k={optimal_k}")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

print(f"\nKüme dağılımı:")
for i in range(optimal_k):
    n = (clusters == i).sum()
    ciddi_oran = y_sample[clusters == i].mean() * 100
    print(f"  Küme {i}: {n:,} kayıt | Ciddi kaza oranı: %{ciddi_oran:.1f}")

# ============================================================
# 5. PCA İLE GÖRSELLEŞTİRME
# ============================================================
print("\nPCA uygulanıyor...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Açıklanan varyans: PC1=%{pca.explained_variance_ratio_[0]*100:.1f}, PC2=%{pca.explained_variance_ratio_[1]*100:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors_cluster = ['#1565C0', '#C62828', '#2E7D32', '#F57F17']

# Sol: Kümelere göre renklendirme
for i in range(optimal_k):
    mask = clusters == i
    ciddi_oran = y_sample[mask].mean() * 100
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=colors_cluster[i], alpha=0.4, s=8,
                    label=f'Küme {i} (%{ciddi_oran:.0f} ciddi)')
axes[0].set_title('K-Means Kümeleri (PCA 2D)', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 (%{pca.explained_variance_ratio_[0]*100:.1f})')
axes[0].set_ylabel(f'PC2 (%{pca.explained_variance_ratio_[1]*100:.1f})')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.2)

# Sağ: Şiddete göre renklendirme
colors_sev = ['#388e3c', '#d32f2f']
labels_sev = ['Hafif', 'Ciddi+Ölümcül']
for i, (color, label) in enumerate(zip(colors_sev, labels_sev)):
    mask = y_sample == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, alpha=0.3, s=8, label=label)
axes[1].set_title('Kaza Şiddetine Göre PCA Dağılımı', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 (%{pca.explained_variance_ratio_[0]*100:.1f})')
axes[1].set_ylabel(f'PC2 (%{pca.explained_variance_ratio_[1]*100:.1f})')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('10_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ PCA kümeleme grafiği kaydedildi")

# ============================================================
# 6. KÜME PROFİL ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("KÜME PROFİL ANALİZİ")
print("=" * 60)

X_df = pd.DataFrame(X_sample, columns=feature_names)
X_df['Cluster'] = clusters
X_df['Severity'] = y_sample

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

key_features = ['Speed_limit', 'Number_of_Vehicles', 'Light_Conditions', 'Road_Surface_Conditions']
key_features = [f for f in key_features if f in X_df.columns]

for ax, feat in zip(axes, key_features):
    cluster_means = X_df.groupby('Cluster')[feat].mean()
    ax.bar(range(optimal_k), cluster_means.values,
           color=colors_cluster[:optimal_k])
    ax.set_title(f'{feat} - Kümelere Göre Ortalama', fontweight='bold')
    ax.set_xlabel('Küme')
    ax.set_ylabel('Ortalama Değer')
    ax.set_xticks(range(optimal_k))
    ax.set_xticklabels([f'Küme {i}' for i in range(optimal_k)])
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('11_cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Küme profil grafiği kaydedildi")

print("\nClustering tamamlandı!")
