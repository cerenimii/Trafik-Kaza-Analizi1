# ============================================================
# TRAFFIC ACCIDENT SEVERITY PREDICTION SYSTEM
# 05 - Zaman Serisi Analizi
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ZAMAN SERİSİ ANALİZİ")
print("=" * 60)

# ============================================================
# 1. VERİYİ YÜKLEME
# ============================================================
df = pd.read_csv('merged_accidents.csv', low_memory=False)

# Tarih ve saat işleme
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Year']    = df['Date'].dt.year
df['Month']   = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Pazartesi
df['Hour']    = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
df['Season']  = df['Month'].map({
    12: 'Kış', 1: 'Kış', 2: 'Kış',
    3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
    6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
    9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
})
df['Severity_Binary'] = df['Accident_Severity'].apply(lambda x: 1 if x in [1, 2] else 0)
df = df.dropna(subset=['Year', 'Hour'])
df['Year'] = df['Year'].astype(int)
df = df[(df['Year'] >= 2005) & (df['Year'] <= 2014)]

print(f"Analiz için: {len(df):,} kayıt ({df['Year'].min()} - {df['Year'].max()})")

# ============================================================
# 2. YILLIK KAZA TRENDİ
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

yearly = df.groupby('Year').agg(
    Toplam_Kaza=('Accident_Severity', 'count'),
    Ciddi_Kaza=('Severity_Binary', 'sum')
).reset_index()
yearly['Ciddi_Oran'] = yearly['Ciddi_Kaza'] / yearly['Toplam_Kaza'] * 100

ax1 = axes[0]
ax2 = ax1.twinx()
ax1.bar(yearly['Year'], yearly['Toplam_Kaza'], color='#1565C0', alpha=0.6, label='Toplam Kaza')
ax2.plot(yearly['Year'], yearly['Ciddi_Oran'], 'r-o', linewidth=2.5, markersize=7, label='Ciddi Oran (%)')
ax1.set_xlabel('Yıl', fontsize=12)
ax1.set_ylabel('Toplam Kaza Sayısı', fontsize=12, color='#1565C0')
ax2.set_ylabel('Ciddi Kaza Oranı (%)', fontsize=12, color='red')
axes[0].set_title('Yıllık Kaza Trendi (2005-2014)', fontsize=14, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Saatlik trend
hourly = df.groupby('Hour').agg(
    Toplam=('Accident_Severity', 'count'),
    Ciddi_Oran=('Severity_Binary', 'mean')
).reset_index()
hourly['Ciddi_Oran'] *= 100

ax3 = axes[1]
ax4 = ax3.twinx()
ax3.bar(hourly['Hour'], hourly['Toplam'], color='#6A1B9A', alpha=0.6, label='Toplam Kaza')
ax4.plot(hourly['Hour'], hourly['Ciddi_Oran'], 'r-o', linewidth=2.5, markersize=6, label='Ciddi Oran (%)')
ax3.set_xlabel('Saat', fontsize=12)
ax3.set_ylabel('Kaza Sayısı', fontsize=12, color='#6A1B9A')
ax4.set_ylabel('Ciddi Kaza Oranı (%)', fontsize=12, color='red')
axes[1].set_title('Günlük Saatlere Göre Kaza Dağılımı', fontsize=14, fontweight='bold')
ax3.set_xticks(range(0, 24))
ax3.axvspan(7, 9, alpha=0.1, color='red', label='Rush Hour')
ax3.axvspan(16, 19, alpha=0.1, color='red')

lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')

plt.tight_layout()
plt.savefig('12_time_series_main.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Ana zaman serisi grafiği kaydedildi")

# ============================================================
# 3. AYLIK VE MEVSİMSEL ANALİZ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

monthly = df.groupby('Month')['Severity_Binary'].agg(['mean', 'count']).reset_index()
monthly['mean'] *= 100
ay_isimleri = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz',
               'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']
bars = axes[0].bar(range(1, 13), monthly['mean'], color='#0288D1')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(ay_isimleri, rotation=45)
axes[0].set_title('Aylara Göre Ciddi Kaza Oranı', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Ciddi Kaza Oranı (%)')
axes[0].grid(alpha=0.3, axis='y')

season_order = ['Kış', 'İlkbahar', 'Yaz', 'Sonbahar']
seasonal = df.groupby('Season')['Severity_Binary'].mean().reindex(season_order) * 100
colors_season = ['#90CAF9', '#A5D6A7', '#FFCC80', '#CE93D8']
axes[1].bar(season_order, seasonal.values, color=colors_season)
axes[1].set_title('Mevsimlere Göre Ciddi Kaza Oranı', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Ciddi Kaza Oranı (%)')
axes[1].grid(alpha=0.3, axis='y')
for i, v in enumerate(seasonal.values):
    axes[1].text(i, v + 0.1, f'%{v:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('13_monthly_seasonal.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Aylık/mevsimsel grafik kaydedildi")

# ============================================================
# 4. ISIL HARİTA: Saat x Haftanın Günü
# ============================================================
gun_isimleri = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
heatmap_data = df.groupby(['DayOfWeek', 'Hour'])['Severity_Binary'].mean().unstack() * 100

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Ciddi Kaza Oranı (%)'},
            yticklabels=gun_isimleri)
ax.set_title('Gün x Saat Ciddi Kaza Oranı Isıl Haritası', fontsize=14, fontweight='bold')
ax.set_xlabel('Saat')
ax.set_ylabel('Haftanın Günü')
plt.tight_layout()
plt.savefig('14_heatmap_day_hour.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Isıl harita kaydedildi")

# ============================================================
# 5. ÖZET BULGULAR
# ============================================================
print("\n" + "=" * 60)
print("ZAMAN SERİSİ ÖZET BULGULARI")
print("=" * 60)

en_riskli_saat = hourly.loc[hourly['Ciddi_Oran'].idxmax(), 'Hour']
en_az_riskli_saat = hourly.loc[hourly['Ciddi_Oran'].idxmin(), 'Hour']
en_cok_kaza_saat = hourly.loc[hourly['Toplam'].idxmax(), 'Hour']
en_riskli_ay = monthly.loc[monthly['mean'].idxmax(), 'Month']
en_riskli_mevsim = seasonal.idxmax()

print(f"En riskli saat   : {en_riskli_saat}:00 (en yüksek ciddi oran)")
print(f"En kalabalık saat: {en_cok_kaza_saat}:00 (en çok kaza)")
print(f"En riskli ay     : {ay_isimleri[en_riskli_ay-1]}")
print(f"En riskli mevsim : {en_riskli_mevsim}")

print("\nZaman serisi analizi tamamlandı!")
