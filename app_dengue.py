"""
🦟 DASHBOARD DENGUE - CESAR & VALLEDUPAR
Streamlit App
Ejecutar: streamlit run app_dengue.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Cesar - Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    /* Estilos generales */
    .stApp {
        background-color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value { font-size: 2.2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }
    .stTabs [data-baseweb="tab"] { 
        font-size: 1rem; 
        font-weight: 600;
        padding: 8px 16px;
    }
    /* Tarjeta de riesgo */
    .risk-card-low {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .risk-card-moderate {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .symptom-badge {
        background: #e2e8f0;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .info-box {
        background: #e6f7ff;
        border-left: 4px solid #1890ff;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff7e6;
        border-left: 4px solid #faad14;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .danger-box {
        background: #fff1f0;
        border-left: 4px solid #f5222d;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# CARGA DE DATOS Y MODELOS CON CACHÉ
# ──────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    """Carga los modelos y artefactos necesarios para predicciones"""
    modelos = {}
    
    # Verificar si existe modelo de Red Neuronal
    if os.path.exists('mejor_modelo_nn.h5'):
        try:
            from tensorflow.keras.models import load_model
            modelos['modelo'] = load_model('mejor_modelo_nn.h5')
            modelos['tipo'] = 'nn'
            modelos['nombre'] = 'Red Neuronal'
            print("✅ Modelo Red Neuronal cargado")
        except Exception as e:
            st.warning(f"No se pudo cargar modelo Red Neuronal: {e}")
            modelos['modelo'] = None
    # Verificar si existe modelo Random Forest/Logistic
    elif os.path.exists('mejor_modelo.pkl'):
        try:
            import joblib
            modelos['modelo'] = joblib.load('mejor_modelo.pkl')
            modelos['tipo'] = 'sklearn'
            modelos['nombre'] = 'Modelo Scikit-learn'
            print("✅ Modelo Scikit-learn cargado")
        except Exception as e:
            st.warning(f"No se pudo cargar modelo: {e}")
            modelos['modelo'] = None
    else:
        modelos['modelo'] = None
    
    # Cargar scaler
    if os.path.exists('scaler.pkl'):
        try:
            import joblib
            modelos['scaler'] = joblib.load('scaler.pkl')
            print("✅ Scaler cargado")
        except:
            modelos['scaler'] = None
    else:
        modelos['scaler'] = None
    
    # Cargar feature columns
    if os.path.exists('feature_columns.pkl'):
        try:
            import joblib
            modelos['feature_columns'] = joblib.load('feature_columns.pkl')
            print("✅ Feature columns cargadas")
        except:
            modelos['feature_columns'] = None
    else:
        modelos['feature_columns'] = None
    
    # Cargar threshold
    if os.path.exists('threshold.json'):
        try:
            with open('threshold.json', 'r') as f:
                threshold_data = json.load(f)
                modelos['threshold'] = threshold_data.get('threshold', 0.5)
                modelos['modelo_mejor'] = threshold_data.get('modelo', 'Desconocido')
            print(f"✅ Threshold cargado: {modelos['threshold']}")
        except:
            modelos['threshold'] = 0.5
    else:
        modelos['threshold'] = 0.5
    
    return modelos

@st.cache_data
def cargar_datos():
    """Carga los datos históricos y predicciones"""
    archivos = {
        'hist_edad':    'historico_edad.csv',
        'hist_estrato': 'historico_estrato.csv',
        'hist_sexo':    'historico_sexo.csv',
        'hist_anual':   'historico_anual.csv',
        'pred_cesar':   'predicciones_cesar.csv',
        'pred_valle':   'predicciones_valledupar.csv',
        'metricas':     'metricas_modelos.csv',
    }
    datos = {}
    for key, fname in archivos.items():
        if os.path.exists(fname):
            datos[key] = pd.read_csv(fname)
        else:
            datos[key] = None
    return datos

# Cargar todo
datos = cargar_datos()
modelos = cargar_modelos()

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg,#1e3a5f,#2563eb);
            padding:30px; border-radius:15px; margin-bottom:20px; text-align:center;'>
    <h1 style='color:white; margin:0; font-size:2.4rem;'>🦟 Sistema de Vigilancia del Dengue</h1>
    <p style='color:#bdd7f7; margin:8px 0 0 0; font-size:1.1rem;'>
        Departamento del Cesar, Colombia · Análisis histórico y predicciones 2026–2030
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# MÉTRICAS GENERALES (KPIs)
# ──────────────────────────────────────────────────────────
if datos['hist_anual'] is not None:
    df_anual = datos['hist_anual']
    total_historico = int(df_anual['total_casos'].sum())
    total_graves    = int(df_anual['casos_graves'].sum())
    pct_graves      = round(total_graves / total_historico * 100, 2) if total_historico else 0
    anio_pico       = int(df_anual.loc[df_anual['total_casos'].idxmax(), 'anio'])

    col1, col2, col3, col4 = st.columns(4)
    for col, valor, label, color in zip(
        [col1, col2, col3, col4],
        [f"{total_historico:,}", f"{total_graves:,}", f"{pct_graves}%", str(anio_pico)],
        ["Total Casos Históricos", "Casos Dengue Grave", "% Dengue Grave", "Año con más Casos"],
        ["#2563eb", "#dc2626", "#d97706", "#059669"]
    ):
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,{color}22,{color}11);
                    border:2px solid {color}; border-radius:12px;
                    padding:18px; text-align:center;'>
            <div style='font-size:2rem; font-weight:800; color:{color};'>{valor}</div>
            <div style='color:#555; font-size:0.85rem; margin-top:4px;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Análisis Histórico",
    "🔮 Predicciones Futuras",
    "🏆 Métricas del Modelo",
    "🩺 Predictor de Dengue",
    "ℹ️ Acerca del Proyecto"
])

# ══════════════════════════════════════════════════════════
# TAB 1: HISTÓRICO (sin cambios, mantengo tu código original)
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Análisis Histórico de Casos</div>", unsafe_allow_html=True)

    # ── Gráfico 1: Casos por año (línea doble) ────────────
    if datos['hist_anual'] is not None:
        df_anual = datos['hist_anual']
        fig_anual = go.Figure()
        fig_anual.add_trace(go.Scatter(
            x=df_anual['anio'], y=df_anual['total_casos'],
            mode='lines+markers', name='Total Casos',
            line=dict(color='#2563eb', width=2.5),
            marker=dict(size=8),
            hovertemplate="Año: %{x}<br>Total: %{y:,}<extra></extra>"
        ))
        fig_anual.add_trace(go.Scatter(
            x=df_anual['anio'], y=df_anual['casos_graves'],
            mode='lines+markers', name='Dengue Grave',
            line=dict(color='#dc2626', width=2.5, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate="Año: %{x}<br>Graves: %{y:,}<extra></extra>"
        ))
        fig_anual.update_layout(
            title='Evolución Anual de Casos de Dengue',
            xaxis_title='Año', yaxis_title='Número de Casos',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            hovermode='x unified', height=420,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_anual.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig_anual.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig_anual, use_container_width=True)

    col_left, col_right = st.columns(2)

    # ── Gráfico 2: Por sexo ───────────────────────────────
    with col_left:
        if datos['hist_sexo'] is not None:
            df_sexo = datos['hist_sexo']
            if 'sexo' not in df_sexo.columns:
                df_sexo['sexo'] = df_sexo.get('sexo__M', pd.Series([0,1])).map({1:'Masculino',0:'Femenino'})

            fig_sexo = go.Figure()
            fig_sexo.add_trace(go.Bar(
                x=df_sexo['sexo'], y=df_sexo['total_casos'],
                name='Total', marker_color='#3b82f6',
                text=df_sexo['total_casos'].apply(lambda x: f"{x:,}"),
                textposition='outside'
            ))
            fig_sexo.add_trace(go.Bar(
                x=df_sexo['sexo'], y=df_sexo['casos_graves'],
                name='Graves', marker_color='#ef4444',
                text=df_sexo['casos_graves'].apply(lambda x: f"{x:,}"),
                textposition='outside'
            ))
            fig_sexo.update_layout(
                title='Casos por Sexo',
                barmode='group', height=380,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_sexo, use_container_width=True)

    # ── Gráfico 3: Por estrato ─────────────────────────────
    with col_right:
        if datos['hist_estrato'] is not None:
            df_est = datos['hist_estrato']
            col_estrato = 'estrato' if 'estrato' in df_est.columns else df_est.columns[0]
            fig_est = px.bar(
                df_est, x=col_estrato, y='total_casos',
                color='casos_graves', color_continuous_scale='RdYlGn_r',
                title='Casos por Estrato Socioeconómico',
                labels={'total_casos': 'Total Casos', col_estrato: 'Estrato', 'casos_graves': 'Graves'},
                text='total_casos'
            )
            fig_est.update_traces(textposition='outside')
            fig_est.update_layout(
                height=380,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_est, use_container_width=True)

    # ── Gráfico 4: Distribución por edad (histogram style) ─
    if datos['hist_edad'] is not None:
        df_edad = datos['hist_edad']
        col_edad = 'edad' if 'edad' in df_edad.columns else df_edad.columns[0]

        # Agrupar por rangos de edad
        df_edad['rango_edad'] = pd.cut(df_edad[col_edad],
            bins=[0, 5, 10, 18, 30, 45, 60, 200],
            labels=['0-5', '6-10', '11-18', '19-30', '31-45', '46-60', '60+'])
        edad_agrupada = df_edad.groupby('rango_edad', observed=True).agg(
            total_casos=('total_casos','sum'),
            casos_graves=('casos_graves','sum')
        ).reset_index()

        fig_edad = go.Figure()
        fig_edad.add_trace(go.Bar(
            x=edad_agrupada['rango_edad'].astype(str),
            y=edad_agrupada['total_casos'],
            name='Total', marker_color='#60a5fa'
        ))
        fig_edad.add_trace(go.Bar(
            x=edad_agrupada['rango_edad'].astype(str),
            y=edad_agrupada['casos_graves'],
            name='Dengue Grave', marker_color='#f87171'
        ))
        fig_edad.update_layout(
            title='Distribución de Casos por Rango de Edad',
            xaxis_title='Rango de Edad', yaxis_title='Casos',
            barmode='group', height=400,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_edad, use_container_width=True)

    # ── Gráfico 5: % graves por año ───────────────────────
    if datos['hist_anual'] is not None:
        fig_pct = px.area(
            df_anual, x='anio', y='pct_graves',
            title='Porcentaje de Dengue Grave por Año (%)',
            labels={'pct_graves': '% Dengue Grave', 'anio': 'Año'},
            color_discrete_sequence=['#f59e0b']
        )
        fig_pct.update_layout(height=350,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pct, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2: PREDICCIONES (mejorado con más visualizaciones)
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Proyecciones a 5 Años (2026–2030)</div>", unsafe_allow_html=True)

    sidebar_region = st.radio("Seleccionar Región:", ["Cesar (general)", "Valledupar", "Comparación"], horizontal=True)

    def plot_predicciones(df_pred, titulo, color_total='#3b82f6', color_grave='#ef4444'):
        if df_pred is None:
            st.warning(f"No se encontró el archivo de predicciones para {titulo}.")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Total de Casos Proyectados',
                'Casos Dengue Grave Proyectados',
                'Crecimiento Anual (%)',
                '% Dengue Grave por Año'
            )
        )
        anio = df_pred['anio'].astype(str)

        fig.add_trace(go.Bar(x=anio, y=df_pred['total_casos'],
            marker_color=color_total, name='Total',
            text=df_pred['total_casos'].apply(lambda x: f"{x:,}"),
            textposition='outside'), row=1, col=1)

        fig.add_trace(go.Bar(x=anio, y=df_pred['casos_graves'],
            marker_color=color_grave, name='Graves',
            text=df_pred['casos_graves'].apply(lambda x: f"{x:,}"),
            textposition='outside'), row=1, col=2)

        if 'crecimiento_pct' in df_pred.columns:
            crec = df_pred.dropna(subset=['crecimiento_pct'])
            colors_crec = ['#22c55e' if v <= 0 else '#ef4444' for v in crec['crecimiento_pct']]
            fig.add_trace(go.Bar(
                x=crec['anio'].astype(str), y=crec['crecimiento_pct'],
                marker_color=colors_crec, name='Crecimiento %',
                text=crec['crecimiento_pct'].round(1).astype(str) + '%',
                textposition='outside'
            ), row=2, col=1)

        if 'pct_graves' in df_pred.columns:
            fig.add_trace(go.Scatter(
                x=anio, y=df_pred['pct_graves'],
                mode='lines+markers+text',
                line=dict(color='#f59e0b', width=2.5),
                marker=dict(size=10),
                text=df_pred['pct_graves'].round(1).astype(str) + '%',
                textposition='top center', name='% Graves'
            ), row=2, col=2)

        fig.update_layout(
            title_text=f"📍 {titulo} — Predicciones 5 Años",
            height=650, showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig, use_container_width=True)

        # Tabla resumen
        st.dataframe(
            df_pred[['anio','total_casos','casos_graves','pct_graves','crecimiento_pct']].rename(columns={
                'anio':'Año','total_casos':'Total Casos','casos_graves':'Graves',
                'pct_graves':'% Graves','crecimiento_pct':'Crecimiento %'
            }),
            use_container_width=True, hide_index=True
        )

    if sidebar_region == "Cesar (general)":
        plot_predicciones(datos['pred_cesar'], "Departamento del Cesar", '#2563eb', '#dc2626')
    elif sidebar_region == "Valledupar":
        plot_predicciones(datos['pred_valle'], "Valledupar", '#7c3aed', '#f97316')
    else:  # Comparación
        st.markdown("#### Comparación: Cesar vs Valledupar")
        if datos['pred_cesar'] is not None and datos['pred_valle'] is not None:
            pc = datos['pred_cesar'].copy()
            pv = datos['pred_valle'].copy()

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=pc['anio'].astype(str), y=pc['total_casos'],
                mode='lines+markers', name='Cesar - Total',
                line=dict(color='#2563eb', width=2.5),
                marker=dict(size=9)
            ))
            fig_comp.add_trace(go.Scatter(
                x=pv['anio'].astype(str), y=pv['total_casos'],
                mode='lines+markers', name='Valledupar - Total',
                line=dict(color='#7c3aed', width=2.5, dash='dash'),
                marker=dict(size=9, symbol='diamond')
            ))
            fig_comp.add_trace(go.Scatter(
                x=pc['anio'].astype(str), y=pc['casos_graves'],
                mode='lines+markers', name='Cesar - Graves',
                line=dict(color='#dc2626', width=2),
                marker=dict(size=8)
            ))
            fig_comp.add_trace(go.Scatter(
                x=pv['anio'].astype(str), y=pv['casos_graves'],
                mode='lines+markers', name='Valledupar - Graves',
                line=dict(color='#f97316', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            fig_comp.update_layout(
                title='Comparación de Proyecciones: Cesar vs Valledupar',
                xaxis_title='Año', yaxis_title='Casos',
                hovermode='x unified', height=450,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Faltan archivos de predicciones. Ejecuta el pipeline de Colab primero.")


# ══════════════════════════════════════════════════════════
# TAB 3: MÉTRICAS DEL MODELO (mejorado)
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Rendimiento de los Modelos de ML</div>", unsafe_allow_html=True)

    if datos['metricas'] is not None:
        df_met = datos['metricas']

        # Mostrar información del modelo actual
        if modelos.get('modelo') is not None:
            st.info(f"🤖 **Modelo activo en el predictor:** {modelos.get('modelo_mejor', 'Desconocido')} | Threshold: {modelos.get('threshold', 0.5)}")

        # Tabla de métricas
        col_fmt = {c: "{:.3f}" for c in ['recall','precision','f1','roc_auc','pr_auc'] if c in df_met.columns}
        st.dataframe(
            df_met.rename(columns={
                'modelo':'Modelo','recall':'Recall','precision':'Precision',
                'f1':'F1-Score','roc_auc':'ROC-AUC','pr_auc':'PR-AUC'
            }),
            use_container_width=True, hide_index=True
        )

        # Gráfico de barras comparativo
        metricas_cols = [c for c in ['recall','precision','f1','roc_auc','pr_auc'] if c in df_met.columns]
        df_melt = df_met.melt(id_vars='modelo', value_vars=metricas_cols,
                               var_name='Métrica', value_name='Valor')
        fig_met = px.bar(
            df_melt, x='Métrica', y='Valor', color='modelo', barmode='group',
            title='Comparación de Métricas por Modelo',
            color_discrete_sequence=['#2563eb','#16a34a','#dc2626'],
            text=df_melt['Valor'].round(3).astype(str)
        )
        fig_met.update_traces(textposition='outside')
        fig_met.update_layout(
            yaxis_range=[0, 1.1], height=450,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_met, use_container_width=True)

        # Highlight mejor modelo por recall
        if 'recall' in df_met.columns:
            mejor = df_met.loc[df_met['recall'].idxmax(), 'modelo']
            st.success(f"🥇 **Mejor modelo por Recall (detección de casos graves):** {mejor}")
    else:
        st.info("Ejecuta el pipeline de Colab para generar metricas_modelos.csv")


# ══════════════════════════════════════════════════════════
# TAB 4: PREDICTOR DE DENGUE (CORREGIDO)
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>🩺 Predictor de Dengue - Evaluación de Riesgo</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    📋 **Instrucciones:** Selecciona los síntomas y características del paciente. 
    El modelo analizará la información y estimará la probabilidad de que el caso sea dengue grave.
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar que los modelos están cargados
    if modelos.get('modelo') is None:
        st.error("""
        ⚠️ **Modelo no disponible**
        
        No se encontró el archivo del modelo entrenado. Para usar el predictor:
        1. Ejecuta el pipeline en Colab primero
        2. Asegúrate de que `mejor_modelo.pkl` o `mejor_modelo_nn.h5` estén en la misma carpeta
        3. También necesitas `scaler.pkl` y `feature_columns.pkl`
        """)
        st.stop()
    
    # Mostrar información del modelo cargado
    st.info(f"🤖 **Modelo activo:** {modelos.get('modelo_mejor', 'Desconocido')} | Threshold: {modelos.get('threshold', 0.5)}")
    
    # Crear columnas para el formulario
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        st.markdown("#### 👤 Datos Demográficos")
        
        # Edad (slider)
        edad = st.slider(
            "Edad del paciente (años)",
            min_value=0, max_value=100, value=30,
            help="La edad es un factor importante en la gravedad del dengue"
        )
        
        # Sexo
        sexo = st.radio(
            "Sexo",
            options=["Femenino", "Masculino"],
            horizontal=True,
            help="Algunos estudios muestran diferencias en la gravedad por sexo"
        )
        sexo_val = 1 if sexo == "Masculino" else 0
        
        # Estrato
        estrato = st.selectbox(
            "Estrato socioeconómico",
            options=[1, 2, 3, 4, 5, 6],
            help="Estrato 1 es el más bajo, estrato 6 el más alto"
        )
        
        st.markdown("#### 📍 Información Adicional")
        
        # Código municipal (simplificado)
        cod_mun = st.selectbox(
            "Municipio",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Valledupar", 2: "Aguachica", 3: "Bosconia", 4: "La Paz", 5: "Otros"}.get(x, "Otros"),
            help="Municipio de residencia del paciente"
        )
    
    with col_form2:
        st.markdown("#### 🤒 Síntomas Clínicos")
        
        st.markdown("**Síntomas comunes:**")
        col_sint1, col_sint2 = st.columns(2)
        
        with col_sint1:
            fiebre = st.checkbox("Fiebre", value=False, help="Temperatura > 38°C")
            cefalea = st.checkbox("Cefalea (dolor de cabeza)", value=False)
            dolrretroo = st.checkbox("Dolor retroocular", value=False)
            malgias = st.checkbox("Malgias (dolores musculares)", value=False)
            artralgia = st.checkbox("Artralgia (dolor articular)", value=False)
            erupcionr = st.checkbox("Erupción cutánea", value=False)
            dolor_abdo = st.checkbox("Dolor abdominal intenso", value=False)
            vomito = st.checkbox("Vómito persistente", value=False)
        
        with col_sint2:
            diarrea = st.checkbox("Diarrea", value=False)
            somnolenci = st.checkbox("Somnolencia", value=False)
            hipotensio = st.checkbox("Hipotensión", value=False)
            hepatomeg = st.checkbox("Hepatomegalia", value=False)
            hem_mucosa = st.checkbox("Hemorragia de mucosas", value=False)
            hipotermia = st.checkbox("Hipotermia", value=False)
            choque = st.checkbox("Choque/Shock", value=False)
            daño_organ = st.checkbox("Daño orgánico", value=False)
    
    # Botón de predicción
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predecir = st.button("🔍 Evaluar Riesgo de Dengue Grave", type="primary", use_container_width=True)
    
    # Realizar predicción
    if predecir:
        with st.spinner("Analizando síntomas y generando predicción..."):
            try:
                # DEBUG: Mostrar las columnas esperadas
                if modelos['feature_columns'] is not None:
                    feature_cols = modelos['feature_columns']
                    # st.write("Columnas esperadas por el modelo:", feature_cols[:10])  # Debug opcional
                else:
                    st.error("No se encontraron las columnas de features")
                    st.stop()
                
                # Inicializar todas las características en 0
                input_dict = {col: 0 for col in feature_cols}
                
                # Mapear inputs del usuario a las columnas correctas
                # IMPORTANTE: Verificar nombres exactos de las columnas
                # Variables demográficas - VERIFICAR NOMBRES EXACTOS
                posibles_edad = ['edad_', 'edad', 'Edad']
                for col in posibles_edad:
                    if col in input_dict:
                        input_dict[col] = edad
                        break
                
                posibles_sexo = ['sexo__M', 'sexo_M', 'sexo_m', 'Sexo']
                for col in posibles_sexo:
                    if col in input_dict:
                        input_dict[col] = sexo_val
                        break
                
                posibles_estrato = ['estrato_', 'estrato', 'Estrato']
                for col in posibles_estrato:
                    if col in input_dict:
                        input_dict[col] = estrato
                        break
                
                posibles_cod_mun = ['cod_mun_r', 'cod_mun', 'municipio']
                for col in posibles_cod_mun:
                    if col in input_dict:
                        input_dict[col] = cod_mun
                        break
                
                # Variables de síntomas - usar nombres exactos
                sintomas_map = {
                    'fiebre': fiebre,
                    'cefalea': cefalea,
                    'dolrretroo': dolrretroo,
                    'malgias': malgias,
                    'artralgia': artralgia,
                    'erupcionr': erupcionr,
                    'dolor_abdo': dolor_abdo,
                    'vomito': vomito,
                    'diarrea': diarrea,
                    'somnolenci': somnolenci,
                    'hipotensio': hipotensio,
                    'hepatomeg': hepatomeg,
                    'hem_mucosa': hem_mucosa,
                    'hipotermia': hipotermia,
                    'choque': choque,
                    'daño_organ': daño_organ
                }
                
                for sintoma, valor in sintomas_map.items():
                    if sintoma in input_dict:
                        input_dict[sintoma] = 1 if valor else 0
                
                # Variables adicionales que pueden estar presentes
                # Establecer valores predeterminados para columnas requeridas
                for col in feature_cols:
                    if col not in input_dict:
                        # Variables de fecha
                        if col.startswith('fec_not_') or col.startswith('fec_con_') or col.startswith('ini_sin_'):
                            input_dict[col] = 2025
                        elif col == 'anio':
                            input_dict[col] = 2025
                        elif col == 'tiene_fec_hos':
                            input_dict[col] = 0
                        elif col in ['pac_hos_', 'tip_cas_']:
                            input_dict[col] = 0
                        else:
                            # Otras variables, mantener 0
                            input_dict[col] = 0
                
                # Crear DataFrame
                input_df = pd.DataFrame([input_dict])
                
                # Asegurar que las columnas estén en el orden correcto
                input_df = input_df[feature_cols]
                
                # DEBUG: Mostrar valores de entrada (opcional, quitar en producción)
                # st.write("Valores de entrada (primeras 10):", input_df.iloc[0][:10].to_dict())
                
                # Escalar los datos
                if modelos['scaler'] is not None:
                    input_scaled = modelos['scaler'].transform(input_df)
                else:
                    input_scaled = input_df.values
                
                # Realizar predicción
                if modelos.get('tipo') == 'nn':
                    proba = modelos['modelo'].predict(input_scaled, verbose=0).flatten()[0]
                else:
                    proba = modelos['modelo'].predict_proba(input_scaled)[0][1]
                
                # Aplicar threshold
                threshold = modelos.get('threshold', 0.5)
                prediccion = proba >= threshold
                
                # Mostrar resultados
                st.markdown("---")
                st.markdown("## 📊 Resultado de la Evaluación")
                
                # Determinar nivel de riesgo basado en probabilidad REAL
                if proba < 0.2:
                    nivel = "BAJO"
                    nivel_color = "risk-card-low"
                    nivel_texto = "Riesgo Bajo"
                    recomendacion_leve = "✅ Síntomas leves. Mantener hidratación y monitorear evolución."
                    recomendacion_grave = ""
                    mostrar_recomendacion_leve = True
                elif proba < 0.5:
                    nivel = "MODERADO"
                    nivel_color = "risk-card-moderate"
                    nivel_texto = "Riesgo Moderado"
                    recomendacion_leve = ""
                    recomendacion_grave = "⚠️ Acudir a consulta médica. Mantener hidratación y reposo. Estar atento a signos de alarma."
                    mostrar_recomendacion_leve = False
                elif proba < 0.8:
                    nivel = "ALTO"
                    nivel_color = "risk-card-high"
                    nivel_texto = "Riesgo Alto"
                    recomendacion_leve = ""
                    recomendacion_grave = "🚨 ACUDIR INMEDIATAMENTE A URGENCIAS. Riesgo significativo de dengue grave."
                    mostrar_recomendacion_leve = False
                else:
                    nivel = "MUY ALTO"
                    nivel_color = "risk-card-high"
                    nivel_texto = "Riesgo Muy Alto"
                    recomendacion_leve = ""
                    recomendacion_grave = "🚨 URGENCIA MÉDICA INMEDIATA. Alto riesgo de complicaciones graves."
                    mostrar_recomendacion_leve = False
                
                # Mostrar medidor de probabilidad
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.markdown(f"""
                    <div class='{nivel_color}'>
                        <div style='font-size:1rem;'>Nivel de Riesgo</div>
                        <div style='font-size:2rem; font-weight:bold;'>{nivel_texto}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metric2:
                    st.markdown(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, #2563eb, #1e40af);'>
                        <div style='font-size:1rem;'>Probabilidad de Dengue Grave</div>
                        <div style='font-size:2.5rem; font-weight:bold;'>{proba*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metric3:
                    resultado = "🔴 DENGUE GRAVE" if prediccion else "🟢 DENGUE (no grave)"
                    st.markdown(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, {("#dc2626" if prediccion else "#16a34a")}, {("#b91c1c" if prediccion else "#15803d")});'>
                        <div style='font-size:1rem;'>Resultado</div>
                        <div style='font-size:1.5rem; font-weight:bold;'>{resultado}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar medidor visual
                st.markdown("#### 📊 Escala de Riesgo")
                
                # Crear gauge chart con la probabilidad REAL
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba * 100,
                    title = {'text': "Probabilidad de Dengue Grave (%)"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#2563eb"},
                        'steps': [
                            {'range': [0, 20], 'color': "#22c55e"},
                            {'range': [20, 50], 'color': "#f59e0b"},
                            {'range': [50, 80], 'color': "#ef4444"},
                            {'range': [80, 100], 'color': "#991b1b"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Recomendaciones - Mostrar según el nivel de riesgo
                st.markdown("#### 💊 Recomendaciones")
                
                # Detectar signos de alarma (síntomas graves seleccionados)
                signos_alarma = []
                if dolor_abdo:
                    signos_alarma.append("⚠️ Dolor abdominal intenso")
                if vomito:
                    signos_alarma.append("⚠️ Vómito persistente")
                if hem_mucosa:
                    signos_alarma.append("⚠️ Hemorragia de mucosas")
                if somnolenci:
                    signos_alarma.append("⚠️ Somnolencia o letargo")
                if hipotensio:
                    signos_alarma.append("⚠️ Hipotensión")
                if choque:
                    signos_alarma.append("🚨 SIGNOS DE CHOQUE - URGENCIA")
                if daño_organ:
                    signos_alarma.append("🚨 DAÑO ORGÁNICO - URGENCIA")
                
                # Mostrar signos de alarma si existen
                if signos_alarma:
                    st.markdown("""
                    <div class='danger-box'>
                        <strong>⚠️ SIGNOS DE ALARMA DETECTADOS</strong><br>
                        Los siguientes síntomas requieren atención médica inmediata:
                    </div>
                    """, unsafe_allow_html=True)
                    for signo in signos_alarma:
                        st.markdown(f"- {signo}")
                
                # Mostrar recomendación según nivel de riesgo (grave o leve)
                if mostrar_recomendacion_leve:
                    st.markdown(f"""
                    <div class='info-box'>
                        <strong>📋 {recomendacion_leve}</strong><br><br>
                        <strong>Recomendaciones generales:</strong><br>
                        • Mantener hidratación (agua, sueros orales)<br>
                        • Reposo absoluto<br>
                        • No automedicarse, especialmente evitar aspirinas o ibuprofeno<br>
                        • Monitorear temperatura cada 4 horas
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='danger-box'>
                        <strong>🚨 {recomendacion_grave}</strong><br><br>
                        <strong>Medidas inmediatas:</strong><br>
                        • ACUDIR AL CENTRO DE SALUD MÁS CERCANO<br>
                        • No automedicarse<br>
                        • Mantener hidratación durante el traslado<br>
                        • No usar aspirinas, ibuprofeno o antiinflamatorios
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mensaje de consulta médica SIEMPRE visible
                st.markdown("""
                <div class='info-box' style='background: #f0f9ff; border-left-color: #0891b2;'>
                    <strong>🏥 IMPORTANTE:</strong> Esta herramienta es un apoyo diagnóstico basado en modelos predictivos. 
                    <strong>No reemplaza la evaluación médica profesional.</strong> Ante cualquier síntoma de alarma, 
                    acuda inmediatamente a un centro de salud.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
                st.info("Verifica que los modelos y artefactos estén correctamente cargados.")
                # Mostrar más información para debugging
                st.write("Detalles del error:", str(e))