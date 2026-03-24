# 🦟 Sistema de Predicción de Dengue — Cesar, Colombia

## Archivos incluidos

| Archivo | Dónde va | Descripción |
|---------|----------|-------------|
| `colab_dengue_pipeline.py` | Google Colab | Pipeline completo ML |
| `app_dengue.py` | Tu PC/servidor | App Streamlit |
| `requirements.txt` | Tu PC/servidor | Dependencias Python |

---

## PASO 1 — Ejecutar el pipeline en Google Colab

1. Abre **Google Colab** (colab.research.google.com)
2. Crea un nuevo notebook
3. En la primera celda, instala dependencias:
   ```python
   !pip install imbalanced-learn scikit-learn pandas numpy matplotlib seaborn plotly openpyxl tensorflow joblib -q
   ```
4. Sube tu archivo `DatasetParaModelar.xlsx` a Colab:
   ```python
   from google.colab import files
   files.upload()  # selecciona DatasetParaModelar.xlsx
   ```
5. **Copia y pega** el contenido completo de `colab_dengue_pipeline.py` en una celda nueva
6. Ejecuta la celda (Shift+Enter o botón ▶)
7. Al finalizar, **descarga** todos los archivos generados:
   ```python
   from google.colab import files
   archivos = [
       'historico_edad.csv', 'historico_estrato.csv',
       'historico_sexo.csv', 'historico_anual.csv',
       'predicciones_cesar.csv', 'predicciones_valledupar.csv',
       'metricas_modelos.csv', 'mejor_modelo.pkl',
       'scaler.pkl', 'feature_columns.pkl'
   ]
   for f in archivos:
       try: files.download(f)
       except: print(f"No encontrado: {f}")
   ```

---

## PASO 2 — Ejecutar la app Streamlit

1. Asegúrate de tener Python 3.9+ instalado
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Coloca **todos los CSV y PKL** descargados de Colab en la misma carpeta que `app_dengue.py`
4. Ejecuta:
   ```bash
   streamlit run app_dengue.py
   ```
5. Se abrirá automáticamente en tu navegador en http://localhost:8501

---

## Estructura de carpetas recomendada

```
proyecto_dengue/
├── app_dengue.py              ← App Streamlit
├── requirements.txt           ← Dependencias
├── historico_edad.csv         ← Generado en Colab
├── historico_estrato.csv      ← Generado en Colab
├── historico_sexo.csv         ← Generado en Colab
├── historico_anual.csv        ← Generado en Colab
├── predicciones_cesar.csv     ← Generado en Colab
├── predicciones_valledupar.csv← Generado en Colab
├── metricas_modelos.csv       ← Generado en Colab
├── mejor_modelo.pkl           ← Generado en Colab
├── scaler.pkl                 ← Generado en Colab
└── feature_columns.pkl        ← Generado en Colab
```

---

## Notas importantes

- **`clasfinal`** y **`nom_eve`** son eliminadas automáticamente (data leakage)
- **`fec_hos`** es eliminada (valores nulos por diseño)
- **SMOTE** se aplica SOLO al set de entrenamiento (evita data leakage)
- El threshold por defecto para predicciones es **0.4** (más sensible para detectar casos graves)
- El modelo prioriza **Recall** sobre Precision (es más costoso perder un caso grave)
- Si `cod_mun_r == 1` no corresponde a Valledupar en tu dataset, ajusta ese valor en la sección 13

---

## Gráficos disponibles en Streamlit

### 📊 Histórico
- Evolución anual total vs graves (línea doble)
- Casos por sexo (barras agrupadas)
- Casos por estrato socioeconómico (barras con color)
- Distribución por rango de edad
- % de dengue grave por año (área)

### 🔮 Predicciones
- Total de casos proyectados por año (barras)
- Casos graves proyectados por año (barras)
- Crecimiento anual % (barras con color semáforo)
- % graves por año (línea con marcadores)
- Comparación Cesar vs Valledupar (líneas múltiples)

### 🏆 Métricas
- Tabla comparativa Recall / Precision / F1 / ROC-AUC / PR-AUC
- Gráfico de barras agrupadas por modelo
