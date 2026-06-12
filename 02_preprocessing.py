# ============================================================
# TRAFFIC ACCIDENT SEVERITY PREDICTION SYSTEM
# 02 - Veri Ön İşleme
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("VERİ ÖN İŞLEME BAŞLIYOR...")
print("=" * 60)

# ============================================================
# 1. VERİYİ YÜKLEME
# ============================================================
df = pd.read_csv('merged_accidents.csv', low_memory=False)
print(f"Yüklenen kayıt: {df.shape[0]:,}")

# ============================================================
# 2. KULLANILACAK KOLONLAR
# ============================================================
target = 'Accident_Severity'

feature_cols = [
    'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit',
    'Day_of_Week', 'Light_Conditions', 'Weather_Conditions',
    'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area',
    'Junction_Detail', 'Junction_Control', 'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities', 'Did_Police_Officer_Attend_Scene_of_Accident'
]

# Sadece mevcut kolonları seç
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"\nKullanılacak özellikler ({len(feature_cols)} adet):\n{feature_cols}")

df = df[feature_cols + [target]].copy()

# ============================================================
# 3. HEDEF DEĞİŞKEN - BINARY
# ============================================================
df['Severity_Binary'] = df[target].apply(lambda x: 1 if x in [1, 2] else 0)
print(f"\nBinary hedef dağılımı:\n{df['Severity_Binary'].value_counts()}")

# ============================================================
# 4. TARİH/SAAT ÖZELLİKLERİ 
# ============================================================
df_raw = pd.read_csv('merged_accidents.csv', low_memory=False)
if 'Date' in df_raw.columns:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], dayfirst=True, errors='coerce')
    df['Year']  = df_raw['Date'].dt.year.values[:len(df)]
    df['Month'] = df_raw['Date'].dt.month.values[:len(df)]
    df['Season'] = df['Month'].map({
        12: 0, 1: 0, 2: 0,  # Kış
        3: 1, 4: 1, 5: 1,   # İlkbahar
        6: 2, 7: 2, 8: 2,   # Yaz
        9: 3, 10: 3, 11: 3  # Sonbahar
    })
    print("✓ Tarih özellikleri çıkarıldı: Year, Month, Season")

if 'Time' in df_raw.columns:
    df['Hour'] = pd.to_datetime(df_raw['Time'], format='%H:%M', errors='coerce').dt.hour.values[:len(df)]
    df['Rush_Hour'] = df['Hour'].apply(lambda x: 1 if x in range(7,10) or x in range(16,20) else 0)
    print("✓ Saat özellikleri çıkarıldı: Hour, Rush_Hour")

# ============================================================
# 5. GEÇERSİZ KODLARI TEMİZLEME (-1, 99)
# ============================================================
print("\nGeçersiz değer temizliği:")
for col in df.select_dtypes(include='number').columns:
    invalid = df[col].isin([-1, 9, 99])
    if invalid.sum() > 0:
        df.loc[invalid, col] = df[col].mode()[0]
        print(f"  {col}: {invalid.sum():,} geçersiz değer düzeltildi")

# ============================================================
# 6. EKSİK DEĞERLERİ DOLDUR
# ============================================================
print("\nEksik değer doldurma:")
for col in df.select_dtypes(include='object').columns:
    n = df[col].isnull().sum()
    if n > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  {col}: {n:,} eksik → mod ile dolduruldu")

for col in df.select_dtypes(include='number').columns:
    n = df[col].isnull().sum()
    if n > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  {col}: {n:,} eksik → medyan ile dolduruldu")

# ============================================================
# 7. AYKIRI DEĞER TEMİZLİĞİ
# ============================================================
before = len(df)
if 'Number_of_Casualties' in df.columns:
    df = df[df['Number_of_Casualties'] < 50]
if 'Number_of_Vehicles' in df.columns:
    df = df[df['Number_of_Vehicles'] < 20]
print(f"\nAykırı değer sonrası: {before:,} → {len(df):,} kayıt")

# ============================================================
# 8. KATEGORİK DEĞİŞKENLERİ ENCODE ETME
# ============================================================
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))
print("\n✓ Label Encoding tamamlandı")

# ============================================================
# 9. FEATURE VE TARGET AYIRMA
# ============================================================
X = df.drop(columns=[target, 'Severity_Binary'])
y = df['Severity_Binary']

print(f"\nFeature boyutu  : {X.shape}")
print(f"Target dağılımı :\n{y.value_counts()}")

# ============================================================
# 10. TRAIN / TEST BÖLME
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nEğitim seti : {X_train.shape[0]:,} kayıt")
print(f"Test seti   : {X_test.shape[0]:,} kayıt")

# ============================================================
# 11. SMOTE - SINIF DENGESİZLİĞİ
# ============================================================
print("\nSMOTE uygulanıyor...")
print(f"SMOTE öncesi  : {pd.Series(y_train).value_counts().to_dict()}")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"SMOTE sonrası : {pd.Series(y_train_sm).value_counts().to_dict()}")

# ============================================================
# 12. KAYDETME
# ============================================================
import joblib

joblib.dump((X_train_sm, X_test, y_train_sm, y_test, X.columns.tolist()), 'preprocessed_data.pkl')
print("\n Ön işleme tamamlandı!")
print("✓ Veriler kaydedildi: preprocessed_data.pkl")
