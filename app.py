
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# 1. Configuración inicial de la página
st.set_page_config(page_title="Dashboard Delitos CABA 2023", layout="wide", page_icon="🚓")

st.title("Análisis de Seguridad Urbana: CABA 2023")
st.markdown("""
Esta aplicación permite explorar interactivamente los incidentes delictivos registrados en la Ciudad Autónoma de Buenos Aires durante 2023. 
Utiliza un modelo de **Machine Learning (K-Means)** para identificar automáticamente las *Zonas Críticas* o Hotspots de inseguridad.
""")

# 2. Carga y limpieza de datos (con caché para optimizar rendimiento)
@st.cache_data
def load_data():
    df = pd.read_csv("delitos_2023.csv")
    
    # Limpieza idéntica a la Fase 3 de la notebook
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    if 'id-mapa' in df.columns:
        df = df.drop(columns=["id-mapa"])
        
    df = df.dropna(subset=['latitud', 'longitud'])
    df = df[(df['latitud'] < -34) & (df['latitud'] > -35)]
    df = df[(df['longitud'] < -58) & (df['longitud'] > -59)]
    
    return df

df_delitos = load_data()

# 3. Carga del Modelo K-Means
@st.cache_resource
def load_model():
    return joblib.load('modelo_kmeans.pkl')

try:
    kmeans = load_model()
    modelo_cargado = True
except FileNotFoundError:
    st.warning("⚠️ No se encontró el archivo 'modelo_kmeans.pkl'. Ejecutando sin predicción de zonas críticas.")
    modelo_cargado = False

# 4. Barra Lateral - Filtros Interactivos
st.sidebar.header("Filtros de Análisis")

# Filtro por Tipo de Delito
tipos_delito = ["Todos"] + list(df_delitos['tipo'].dropna().unique())
tipo_seleccionado = st.sidebar.selectbox("Seleccione el Tipo de Delito:", tipos_delito)

# Filtro por Franja Horaria
franjas = ["Todas"] + list(np.sort(df_delitos['franja'].dropna().unique()))
franja_seleccionada = st.sidebar.selectbox("Seleccione la Franja Horaria:", franjas)

# Aplicar filtros al dataframe
df_filtrado = df_delitos.copy()
if tipo_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['tipo'] == tipo_seleccionado]
if franja_seleccionada != "Todas":
    df_filtrado = df_filtrado[df_filtrado['franja'] == franja_seleccionada]

# 5. Aplicar predicción del modelo al dataset filtrado
if modelo_cargado and not df_filtrado.empty:
    X_filtrado = df_filtrado[['longitud', 'latitud']]
    df_filtrado['zona_critica_cluster'] = kmeans.predict(X_filtrado)

# 6. Panel de Métricas Principales (KPIs)
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total de Registros (Filtrados)", value=f"{len(df_filtrado):,}")
with col2:
    barrio_comun = df_filtrado['barrio'].mode()[0] if not df_filtrado.empty else "N/A"
    st.metric(label="Barrio Más Afectado", value=barrio_comun)
with col3:
    delito_comun = df_filtrado['tipo'].mode()[0] if not df_filtrado.empty else "N/A"
    st.metric(label="Tipo de Delito Frecuente", value=delito_comun)

# 7. Renderizado del Mapa Geoespacial
st.markdown("### Mapa Geoespacial de Incidentes")
if not df_filtrado.empty:
    # Renombramos las columnas temporalmente para usar st.map nativo de Streamlit
    map_data = df_filtrado[['latitud', 'longitud']].rename(columns={'latitud': 'lat', 'longitud': 'lon'})
    st.map(map_data)
else:
    st.info("No hay datos para mostrar con los filtros seleccionados.")

# 8. Gráficos y Visualizaciones Estadísticas (EDA)
st.markdown("---")
st.markdown("### Análisis Estadístico")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**Top 10 Barrios con más incidentes**")
    top_barrios = df_filtrado['barrio'].value_counts().head(10)
    if not top_barrios.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_barrios.values, y=top_barrios.index, palette="viridis", ax=ax)
        ax.set_xlabel("Cantidad de Delitos")
        ax.set_ylabel("Barrio")
        st.pyplot(fig)
    else:
        st.write("Sin datos.")

with col_chart2:
    st.markdown("**Distribución por Tipo de Delito**")
    if tipo_seleccionado == "Todos":
        top_delitos = df_filtrado['tipo'].value_counts().head(10)
        if not top_delitos.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(x=top_delitos.index, y=top_delitos.values, palette="magma", ax=ax2)
            ax2.set_ylabel("Cantidad")
            ax2.set_xlabel("Tipo de Delito")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        else:
            st.write("Sin datos.")
    else:
        st.info(f"Filtro aplicado: Viendo únicamente '{tipo_seleccionado}'. Cambie el filtro a 'Todos' para ver la distribución completa.")

# 9. Visualización del Modelo K-Means
st.markdown("---")
st.markdown("### Zonas Críticas Predichas (K-Means)")
if modelo_cargado and not df_filtrado.empty:
    fig_kmeans, ax_kmeans = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='longitud', y='latitud',
        hue='zona_critica_cluster',
        palette='Set1',
        data=df_filtrado,
        alpha=0.5, s=15, legend='full', ax=ax_kmeans
    )

    # Graficar los centroides originales del modelo
    ax_kmeans.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200, marker='X', c='black', label='Hotspots (Centroides)'
    )

    ax_kmeans.set_title(f"Agrupamiento K-Means interactivo (n_clusters={kmeans.n_clusters})")
    ax_kmeans.set_xlabel("Longitud")
    ax_kmeans.set_ylabel("Latitud")
    ax_kmeans.legend()
    st.pyplot(fig_kmeans)
else:
    st.write("El modelo no está disponible o no hay datos para procesar.")
