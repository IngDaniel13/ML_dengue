# ============================================================
# 🦟 PIPELINE DENGUE - CESAR, COLOMBIA
# Google Colab - Listo para copiar y pegar
# ============================================================

# ──────────────────────────────────────────
# INSTALACIONES (ejecutar primero)
# ──────────────────────────────────────────
# !pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn plotly openpyxl tensorflow joblib -q

# ============================================================
# SECCIÓN 1: IMPORTACIONES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             recall_score, f1_score, precision_score)
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

import joblib

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print(" Librerías importadas correctamente")

# ============================================================
# SECCIÓN 2: CARGA DE DATOS
# ============================================================
# Sube el archivo en Colab con:
# from google.colab import files; files.upload()

df = pd.read_excel('DatasetParaModelar.xlsx')

print("="*60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nTipos de datos:\n{df.dtypes.value_counts()}")
print(f"\nDistribución de dengue_grave:\n{df['dengue_grave'].value_counts()}")
print(f"\nDesbalance: {(df['dengue_grave']==1).sum()} graves vs {(df['dengue_grave']==0).sum()} no graves")
print(f"Proporción graves: {df['dengue_grave'].mean()*100:.2f}%")

# ============================================================
# SECCIÓN 3: FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print(" FEATURE ENGINEERING")
print("="*60)

# Columnas a eliminar (data leakage + nulos + irrelevantes)
cols_to_drop = ['fec_hos', 'nom_eve', 'cod_eve', 'clasfinal']
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"Columnas eliminadas (leakage/irrelevantes): {cols_to_drop}")

# Variables de fecha
date_cols = ['fec_not', 'fec_con_', 'ini_sin_']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_anio'] = df[col].dt.year
        df[f'{col}_mes']  = df[col].dt.month
        df.drop(columns=[col], inplace=True)

# Variable principal de año (para predicciones futuras)
if 'fec_not_anio' in df.columns:
    df['anio'] = df['fec_not_anio']
elif 'anio' not in df.columns:
    print(" No se encontró columna de año. Revisa los nombres de fecha.")

print(f"Shape después de feature engineering: {df.shape}")
print(f"Columnas: {list(df.columns)}")

# ============================================================
# SECCIÓN 4: ANÁLISIS EXPLORATORIO → CSVs para Streamlit
# ============================================================
print("\n" + "="*60)
print(" ANÁLISIS EXPLORATORIO - Generando CSVs")
print("="*60)

# Helper para guardar
def guardar_csv(df_agg, nombre):
    df_agg.to_csv(nombre, index=False)
    print(f" Guardado: {nombre}")

# 4A. Histórico por edad
if 'edad_' in df.columns:
    hist_edad = (df.groupby('edad_')
                   .agg(total_casos=('dengue_grave','count'),
                        casos_graves=('dengue_grave','sum'))
                   .reset_index()
                   .rename(columns={'edad_':'edad'}))
    hist_edad['pct_graves'] = (hist_edad['casos_graves'] / hist_edad['total_casos'] * 100).round(2)
    guardar_csv(hist_edad, 'historico_edad.csv')

# 4B. Histórico por estrato
if 'estrato_' in df.columns:
    hist_estrato = (df.groupby('estrato_')
                      .agg(total_casos=('dengue_grave','count'),
                           casos_graves=('dengue_grave','sum'))
                      .reset_index()
                      .rename(columns={'estrato_':'estrato'}))
    hist_estrato['pct_graves'] = (hist_estrato['casos_graves'] / hist_estrato['total_casos'] * 100).round(2)
    guardar_csv(hist_estrato, 'historico_estrato.csv')

# 4C. Histórico por sexo
if 'sexo__M' in df.columns:
    hist_sexo = (df.groupby('sexo__M')
                   .agg(total_casos=('dengue_grave','count'),
                        casos_graves=('dengue_grave','sum'))
                   .reset_index())
    hist_sexo['sexo'] = hist_sexo['sexo__M'].map({1:'Masculino', 0:'Femenino'})
    hist_sexo['pct_graves'] = (hist_sexo['casos_graves'] / hist_sexo['total_casos'] * 100).round(2)
    guardar_csv(hist_sexo, 'historico_sexo.csv')

# 4D. Histórico anual
if 'anio' in df.columns:
    hist_anual = (df.groupby('anio')
                    .agg(total_casos=('dengue_grave','count'),
                         casos_graves=('dengue_grave','sum'))
                    .reset_index())
    hist_anual['pct_graves'] = (hist_anual['casos_graves'] / hist_anual['total_casos'] * 100).round(2)
    anio_max = hist_anual.loc[hist_anual['total_casos'].idxmax(), 'anio']
    print(f"  📌 Año con más casos: {anio_max}")
    guardar_csv(hist_anual, 'historico_anual.csv')

# ============================================================
# SECCIÓN 5: DEFINICIÓN DE X, y
# ============================================================
print("\n" + "="*60)
print("DEFINICIÓN DE VARIABLES")
print("="*60)

TARGET = 'dengue_grave'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Eliminar columnas no numéricas residuales
non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric:
    print(f"  Columnas no numéricas eliminadas: {non_numeric}")
    X.drop(columns=non_numeric, inplace=True)

# Imputar nulos con mediana
X.fillna(X.median(), inplace=True)

print(f"X shape: {X.shape}")
print(f"y distribución:\n{y.value_counts()}")

# ============================================================
# SECCIÓN 6: SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"  Train graves: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Test  graves: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ============================================================
# SECCIÓN 7: SMOTE (solo en entrenamiento → evita data leakage)
# ============================================================
print("\n" + "="*60)
print("  SMOTE - Balanceo de clases")
print("="*60)
print(" SMOTE se aplica SOLO al set de entrenamiento.")
print("      Aplicarlo al test contaminaría la evaluación (data leakage).")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  Antes SMOTE: {y_train.value_counts().to_dict()}")
print(f"  Después SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# ============================================================
# SECCIÓN 8: ESCALADO
# ============================================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# SECCIÓN 9: MODELOS
# ============================================================
print("\n" + "="*60)
print(" ENTRENAMIENTO DE MODELOS")
print("="*60)

results = {}

# ── 9A. RANDOM FOREST ──────────────────────────────────────
print("\n[1/3] Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
rf_proba = rf.predict_proba(X_test)[:,1]
rf_pred  = rf.predict(X_test)
results['Random Forest'] = {
    'model': rf, 'proba': rf_proba, 'pred': rf_pred,
    'recall':    recall_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'f1':        f1_score(y_test, rf_pred),
    'roc_auc':   roc_auc_score(y_test, rf_proba),
    'pr_auc':    average_precision_score(y_test, rf_proba),
}
print(f"  RF  | Recall={results['Random Forest']['recall']:.3f} | F1={results['Random Forest']['f1']:.3f} | ROC-AUC={results['Random Forest']['roc_auc']:.3f}")

# ── 9B. REGRESIÓN LOGÍSTICA ────────────────────────────────
print("[2/3] Regresión Logística...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train_res)
lr_proba = lr.predict_proba(X_test_sc)[:,1]
lr_pred  = lr.predict(X_test_sc)
results['Regresión Logística'] = {
    'model': lr, 'proba': lr_proba, 'pred': lr_pred,
    'recall':    recall_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'f1':        f1_score(y_test, lr_pred),
    'roc_auc':   roc_auc_score(y_test, lr_proba),
    'pr_auc':    average_precision_score(y_test, lr_proba),
}
print(f"  LR  | Recall={results['Regresión Logística']['recall']:.3f} | F1={results['Regresión Logística']['f1']:.3f} | ROC-AUC={results['Regresión Logística']['roc_auc']:.3f}")

# ── 9C. RED NEURONAL ───────────────────────────────────────
print("[3/3] Red Neuronal (Keras)...")
n_features = X_train_sc.shape[1]
nn = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Class weights para Keras
neg, pos = np.bincount(y_train_res.astype(int))
class_weight_nn = {0: 1.0, 1: neg/pos}

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_nn = nn.fit(
    X_train_sc, y_train_res,
    epochs=50, batch_size=256,
    validation_split=0.1,
    class_weight=class_weight_nn,
    callbacks=[es], verbose=0
)
nn_proba = nn.predict(X_test_sc, verbose=0).flatten()
nn_pred  = (nn_proba >= 0.5).astype(int)
results['Red Neuronal'] = {
    'model': nn, 'proba': nn_proba, 'pred': nn_pred,
    'recall':    recall_score(y_test, nn_pred),
    'precision': precision_score(y_test, nn_pred),
    'f1':        f1_score(y_test, nn_pred),
    'roc_auc':   roc_auc_score(y_test, nn_proba),
    'pr_auc':    average_precision_score(y_test, nn_proba),
}
print(f"  NN  | Recall={results['Red Neuronal']['recall']:.3f} | F1={results['Red Neuronal']['f1']:.3f} | ROC-AUC={results['Red Neuronal']['roc_auc']:.3f}")

# ============================================================
# SECCIÓN 10: MÉTRICAS Y VISUALIZACIONES
# ============================================================
print("\n" + "="*60)
print(" MÉTRICAS DETALLADAS")
print("="*60)

for nombre, r in results.items():
    print(f"\n{'─'*40}")
    print(f" {nombre}")
    print(classification_report(y_test, r['pred'], target_names=['Dengue','Dengue Grave']))
    cm = confusion_matrix(y_test, r['pred'])
    print(f"Matriz de confusión:\n{cm}")

# ── Curvas ROC ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['steelblue', 'darkorange', 'forestgreen']
nombres = list(results.keys())

for ax, (nombre, r), color in zip(axes, results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r['proba'])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {r['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_title(f"Curva ROC - {nombre}", fontsize=13, fontweight='bold')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Curvas ROC por Modelo', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('curvas_roc.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Curvas Precision-Recall ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (nombre, r), color in zip(axes, results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, r['proba'])
    ax.plot(rec, prec, color=color, lw=2, label=f"AP = {r['pr_auc']:.3f}")
    ax.set_title(f"PR Curve - {nombre}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('Curvas Precision-Recall por Modelo', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('curvas_pr.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Distribución de clases ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
clases = ['Dengue (0)', 'Dengue Grave (1)']
conteos = y.value_counts().sort_index().values
axes[0].bar(clases, conteos, color=['steelblue','crimson'], edgecolor='black')
axes[0].set_title('Distribución Original de Clases', fontweight='bold')
axes[0].set_ylabel('Cantidad')
for i, v in enumerate(conteos):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

conteos_smote = pd.Series(y_train_res).value_counts().sort_index().values
axes[1].bar(clases, conteos_smote, color=['steelblue','crimson'], edgecolor='black', alpha=0.8)
axes[1].set_title('Distribución después de SMOTE (Train)', fontweight='bold')
axes[1].set_ylabel('Cantidad')
for i, v in enumerate(conteos_smote):
    axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('distribucion_clases.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SECCIÓN 11: AJUSTE DE THRESHOLD
# ============================================================
print("\n" + "="*60)
print(" AJUSTE DE THRESHOLD")
print("="*60)

thresholds = [0.3, 0.4, 0.5]
best_model_name = None
best_recall = 0

for nombre, r in results.items():
    print(f"\n🔹 {nombre}")
    for t in thresholds:
        pred_t = (r['proba'] >= t).astype(int)
        rec = recall_score(y_test, pred_t)
        f1  = f1_score(y_test, pred_t)
        print(f"  Threshold {t}: Recall={rec:.3f} | F1={f1:.3f}")

# ============================================================
# SECCIÓN 12: SELECCIÓN DEL MEJOR MODELO
# ============================================================
print("\n" + "="*60)
print(" SELECCIÓN DEL MEJOR MODELO")
print("="*60)

df_metricas = pd.DataFrame([
    {
        'modelo': nombre,
        'recall':    r['recall'],
        'precision': r['precision'],
        'f1':        r['f1'],
        'roc_auc':   r['roc_auc'],
        'pr_auc':    r['pr_auc'],
    }
    for nombre, r in results.items()
])
df_metricas['score_combinado'] = df_metricas['recall']*0.6 + df_metricas['f1']*0.4

print(df_metricas.to_string(index=False))
df_metricas.to_csv('metricas_modelos.csv', index=False)
print("metricas_modelos.csv guardado")

mejor = df_metricas.loc[df_metricas['score_combinado'].idxmax(), 'modelo']
print(f"\n MEJOR MODELO: {mejor}")
print("   Justificación: Prioridad al Recall (detectar casos graves) + F1 como control de balance.")

mejor_modelo = results[mejor]['model']
mejor_proba  = results[mejor]['proba']

# ============================================================
# SECCIÓN 13: PREDICCIONES A FUTURO (5 AÑOS)
# ============================================================
print("\n" + "="*60)
print(" PREDICCIONES A FUTURO")
print("="*60)

if 'anio' not in df.columns:
    print("  Variable 'anio' no encontrada. Revisa el feature engineering.")
else:
    anio_ultimo = int(df['anio'].max())
    anios_futuros = list(range(anio_ultimo + 1, anio_ultimo + 6))
    print(f"  Años históricos: {sorted(df['anio'].unique())}")
    print(f"  Años predichos:  {anios_futuros}")

    def generar_futuro(df_base, anios, n_por_anio=None):
        """Simula datos futuros tomando la distribución del último año disponible."""
        ultimo_anio = df_base['anio'].max()
        base = df_base[df_base['anio'] == ultimo_anio].copy()
        if n_por_anio is None:
            n_por_anio = len(base)
        frames = []
        for a in anios:
            sample = base.sample(n=min(n_por_anio, len(base)), replace=True, random_state=a)
            sample = sample.copy()
            sample['anio'] = a
            # Actualizar columnas derivadas de año
            for col in sample.columns:
                if '_anio' in col:
                    sample[col] = a
            frames.append(sample)
        return pd.concat(frames, ignore_index=True)

    # ── CESAR (general) ────────────────────────────────────
    df_futuro_cesar = generar_futuro(df.drop(columns=[TARGET]), anios_futuros)
    df_futuro_cesar.fillna(df.drop(columns=[TARGET]).median(), inplace=True)
    non_num = df_futuro_cesar.select_dtypes(exclude=['number']).columns
    df_futuro_cesar.drop(columns=non_num, inplace=True)
    df_futuro_cesar = df_futuro_cesar[X.columns]

    if mejor in ['Regresión Logística', 'Red Neuronal']:
        X_fut_cesar = scaler.transform(df_futuro_cesar)
    else:
        X_fut_cesar = df_futuro_cesar.values

    if mejor == 'Red Neuronal':
        proba_fut_cesar = mejor_modelo.predict(X_fut_cesar, verbose=0).flatten()
    else:
        proba_fut_cesar = mejor_modelo.predict_proba(X_fut_cesar)[:,1]

    df_futuro_cesar['dengue_grave_pred'] = (proba_fut_cesar >= 0.4).astype(int)
    df_futuro_cesar['anio'] = df_futuro_cesar['anio'].astype(int)

    pred_cesar = (df_futuro_cesar.groupby('anio')
                                  .agg(total_casos=('dengue_grave_pred','count'),
                                       casos_graves=('dengue_grave_pred','sum'))
                                  .reset_index())
    pred_cesar['crecimiento_pct'] = pred_cesar['total_casos'].pct_change().mul(100).round(2)
    pred_cesar['pct_graves']      = (pred_cesar['casos_graves'] / pred_cesar['total_casos'] * 100).round(2)
    print(f"\nCESAR - Predicciones:\n{pred_cesar}")
    pred_cesar.to_csv('predicciones_cesar.csv', index=False)
    print("  predicciones_cesar.csv guardado")

    # ── VALLEDUPAR ─────────────────────────────────────────
    if 'cod_mun_r' in df.columns:
        df_valledupar = df[df['cod_mun_r'] == 1].copy()
        print(f"\n  Registros Valledupar: {len(df_valledupar)}")
    else:
        print("  'cod_mun_r' no encontrada. Usando muestra del 30% como proxy.")
        df_valledupar = df.sample(frac=0.3, random_state=42)

    df_fut_valle = generar_futuro(df_valledupar.drop(columns=[TARGET], errors='ignore'), anios_futuros)
    df_fut_valle.fillna(df.drop(columns=[TARGET]).median(), inplace=True)
    non_num = df_fut_valle.select_dtypes(exclude=['number']).columns
    df_fut_valle.drop(columns=non_num, inplace=True)
    df_fut_valle = df_fut_valle.reindex(columns=X.columns, fill_value=0)

    if mejor in ['Regresión Logística', 'Red Neuronal']:
        X_fut_valle = scaler.transform(df_fut_valle)
    else:
        X_fut_valle = df_fut_valle.values

    if mejor == 'Red Neuronal':
        proba_fut_valle = mejor_modelo.predict(X_fut_valle, verbose=0).flatten()
    else:
        proba_fut_valle = mejor_modelo.predict_proba(X_fut_valle)[:,1]

    df_fut_valle['dengue_grave_pred'] = (proba_fut_valle >= 0.4).astype(int)

    pred_valle = (df_fut_valle.groupby('anio')
                               .agg(total_casos=('dengue_grave_pred','count'),
                                    casos_graves=('dengue_grave_pred','sum'))
                               .reset_index())
    pred_valle['crecimiento_pct'] = pred_valle['total_casos'].pct_change().mul(100).round(2)
    pred_valle['pct_graves']      = (pred_valle['casos_graves'] / pred_valle['total_casos'] * 100).round(2)
    print(f"\n VALLEDUPAR - Predicciones:\n{pred_valle}")
    pred_valle.to_csv('predicciones_valledupar.csv', index=False)
    print("  predicciones_valledupar.csv guardado")

# ============================================================
# SECCIÓN 14: GUARDAR MODELO
# ============================================================
print("\n" + "="*60)
print(" GUARDANDO MODELO")
print("="*60)

if mejor != 'Red Neuronal':
    joblib.dump(mejor_modelo, 'mejor_modelo.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"   mejor_modelo.pkl ({mejor})")
    print(f"   scaler.pkl")
else:
    mejor_modelo.save('mejor_modelo_nn.h5')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"   mejor_modelo_nn.h5 (Red Neuronal)")
    print(f"   scaler.pkl")

joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print(f"   feature_columns.pkl")

print("\n" + "="*60)
print(" PIPELINE COMPLETADO EXITOSAMENTE")
print("="*60)
print("""
Archivos generados:
  📁 CSVs históricos:
     - historico_edad.csv
     - historico_estrato.csv
     - historico_sexo.csv
     - historico_anual.csv
  📁 CSVs predicciones:
     - predicciones_cesar.csv
     - predicciones_valledupar.csv
     - metricas_modelos.csv
  📁 Modelos:
     - mejor_modelo.pkl  (o mejor_modelo_nn.h5)
     - scaler.pkl
     - feature_columns.pkl
  📁 Gráficos:
     - curvas_roc.png
     - curvas_pr.png
     - distribucion_clases.png
""")
