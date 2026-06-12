# ============================================================
# TRAFFIC ACCIDENT SEVERITY PREDICTION SYSTEM
# 01 - Veri Yükleme ve Keşifsel Veri Analizi (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ============================================================
# 1. VERİYİ YÜKLE VE BİRLEŞTİR
# ============================================================
print("=" * 60)
print("VERİ YÜKLENİYOR...")
print("=" * 60)

df1 = pd.read_csv('accidents_2005_to_2007.csv', encoding='latin1')
df2 = pd.read_csv('accidents_2009_to_2011.csv', encoding='latin1')
df3 = pd.read_csv('accidents_2012_to_2014.csv', encoding='latin1')

# Ortak kolonları bul
common_cols = list(set(df1.columns) & set(df2.columns) & set(df3.columns))
df = pd.concat([df1[common_cols], df2[common_cols], df3[common_cols]], ignore_index=True)

print(f"Toplam kayıt sayısı : {df.shape[0]:,}")
print(f"Toplam kolon sayısı : {df.shape[1]}")
print(f"\nKolonlar:\n{list(df.columns)}")

# ============================================================
# 2. HEDEF DEĞİŞKEN KONTROLÜ
# ============================================================
print("\n" + "=" * 60)
print("HEDEF DEĞİŞKEN: Accident_Severity")
print("=" * 60)
print(df['Accident_Severity'].value_counts())
print("\n1 = Ölümcül | 2 = Ciddi | 3 = Hafif")

# Binary hedef: 1 = Ciddi+Ölümcül, 0 = Hafif
df['Severity_Binary'] = df['Accident_Severity'].apply(lambda x: 1 if x in [1, 2] else 0)
print(f"\nBinary dağılım:\n{df['Severity_Binary'].value_counts()}")
print(f"\nCiddi+Ölümcül oranı: %{df['Severity_Binary'].mean()*100:.1f}")

# ============================================================
# 3. GENEL İSTATİSTİKLER
# ============================================================
print("\n" + "=" * 60)
print("TANIMLAYICI İSTATİSTİKLER")
print("=" * 60)

num_cols = ['Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit']
num_cols = [c for c in num_cols if c in df.columns]
print(df[num_cols].describe().round(2))

# ============================================================
# 4. EKSİK DEĞER ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("EKSİK DEĞER ANALİZİ")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Eksik Sayı': missing, 'Oran (%)': missing_pct})
print(missing_df[missing_df['Eksik Sayı'] > 0].sort_values('Oran (%)', ascending=False))

# ============================================================
# 5. GÖRSELLEŞTİRMELER
# ============================================================

# 5.1 Hedef Değişken Dağılımı
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

severity_counts = df['Accident_Severity'].value_counts().sort_index()
axes[0].bar(['Ölümcül (1)', 'Ciddi (2)', 'Hafif (3)'],
            severity_counts.values,
            color=['#d32f2f', '#f57c00', '#388e3c'])
axes[0].set_title('Kaza Şiddeti Dağılımı (3 Sınıf)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Kaza Sayısı')
for i, v in enumerate(severity_counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

binary_counts = df['Severity_Binary'].value_counts()
axes[1].pie(binary_counts.values,
            labels=['Hafif (0)', 'Ciddi+Ölümcül (1)'],
            colors=['#388e3c', '#d32f2f'],
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Binary Hedef Değişken Dağılımı', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Grafik kaydedildi: 01_target_distribution.png")

# 5.2 Hız Limiti ve Şiddet İlişkisi
if 'Speed_limit' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    speed_severity = df.groupby('Speed_limit')['Severity_Binary'].mean() * 100
    ax.bar(speed_severity.index, speed_severity.values, color='#1565C0')
    ax.set_xlabel('Hız Limiti (mph)')
    ax.set_ylabel('Ciddi Kaza Oranı (%)')
    ax.set_title('Hız Limitine Göre Ciddi Kaza Oranı', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_speed_vs_severity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Grafik kaydedildi: 02_speed_vs_severity.png")

# 5.3 Araç Sayısı ve Şiddet
if 'Number_of_Vehicles' in df.columns:
    df_filtered = df[df['Number_of_Vehicles'] < 10]
    fig, ax = plt.subplots(figsize=(12, 5))
    veh_severity = df_filtered.groupby('Number_of_Vehicles')['Severity_Binary'].mean() * 100
    ax.bar(veh_severity.index, veh_severity.values, color='#6A1B9A')
    ax.set_xlabel('Araç Sayısı')
    ax.set_ylabel('Ciddi Kaza Oranı (%)')
    ax.set_title('Araç Sayısına Göre Ciddi Kaza Oranı', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_vehicles_vs_severity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Grafik kaydedildi: 03_vehicles_vs_severity.png")

# 5.4 Kentsel/Kırsal ve Şiddet
if 'Urban_or_Rural_Area' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    urban_severity = df.groupby('Urban_or_Rural_Area')['Severity_Binary'].mean() * 100
    ax.bar(['Kentsel (1)', 'Kırsal (2)'], urban_severity.values, color=['#0288D1', '#558B2F'])
    ax.set_ylabel('Ciddi Kaza Oranı (%)')
    ax.set_title('Kentsel/Kırsal Alana Göre Ciddi Kaza Oranı', fontsize=14, fontweight='bold')
    for i, v in enumerate(urban_severity.values):
        ax.text(i, v + 0.3, f'%{v:.1f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_urban_rural_severity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Grafik kaydedildi: 04_urban_rural_severity.png")

print("\n✅ EDA tamamlandı!")
df.to_csv('merged_accidents.csv', index=False)
print("✓ Birleştirilmiş veri kaydedildi: merged_accidents.csv")
