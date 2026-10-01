import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de países - HELP International", page_icon="🌍", layout="wide")

FEATURES = ["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]
LABELS = ["Necesidad alta", "Intermedio", "Desarrollado"]
DESCRIPTIONS = {
    "child_mort": "Mortalidad de menores de 5 años por cada 1000 nacidos vivos",
    "exports": "Exportaciones de bienes y servicios (% del PIB per cápita)",
    "health": "Gasto total en salud (% del PIB per cápita)",
    "imports": "Importaciones de bienes y servicios (% del PIB per cápita)",
    "income": "Ingreso neto por persona",
    "inflation": "Tasa anual de crecimiento/inflación del PIB",
    "life_expec": "Esperanza de vida (años)",
    "total_fer": "Número estimado de hijos por mujer",
    "gdpp": "PIB per cápita",
}

@st.cache_data(show_spinner=False)
def cargar_datos():
    df = pd.read_csv("Country-data.csv")
    faltantes = [c for c in ["country"] + FEATURES if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en Country-data.csv: {', '.join(faltantes)}")
    if df[FEATURES].isnull().any().any():
        raise ValueError("El dataset contiene valores nulos en las variables usadas por el modelo.")
    return df

@st.cache_resource(show_spinner=False)
def entrenar_modelo(df):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    etiquetas_raw = kmeans.fit_predict(X_scaled)

    # Los IDs 0,1,2 de K-Means son arbitrarios. Se ordenan por la mediana de gdpp
    # para obtener nombres estables e interpretables.
    aux = X.copy()
    aux["cluster_raw"] = etiquetas_raw
    orden = aux.groupby("cluster_raw")["gdpp"].median().sort_values().index.tolist()
    raw_a_rango = {cluster_id: rango for rango, cluster_id in enumerate(orden)}
    rangos = np.array([raw_a_rango[x] for x in etiquetas_raw])

    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(X_scaled)
    silueta = silhouette_score(X_scaled, etiquetas_raw)
    return scaler, kmeans, pca, raw_a_rango, rangos, coords_pca, silueta

try:
    df = cargar_datos()
    scaler, kmeans, pca, raw_a_rango, rangos, coords_pca, silueta = entrenar_modelo(df)
except Exception as e:
    st.error("No se pudo iniciar la aplicación. Verifica que Country-data.csv esté en la misma carpeta que app.py.")
    st.exception(e)
    st.stop()

df_app = df.copy()
df_app["grupo"] = [LABELS[r] for r in rangos]

st.title("🌍 Segmentación de países para HELP International")
st.write("Aplicación de apoyo para priorizar países según indicadores socioeconómicos y de salud mediante **K-Means (k=3)**.")
st.info("Uso académico: la clasificación es relativa al dataset y no reemplaza una evaluación humanitaria con información oficial y actualizada.")

with st.sidebar:
    st.header("Consulta")
    modo = st.radio("Modo de evaluación", ["Seleccionar país existente", "Ingresar país nuevo"])
    st.divider()
    st.metric("Países analizados", len(df))
    st.metric("Coeficiente de silueta", f"{silueta:.3f}")
    st.caption("K-Means: k=3, random_state=42, n_init=10; variables estandarizadas con StandardScaler.")

if modo == "Seleccionar país existente":
    pais = st.selectbox("Selecciona un país", sorted(df["country"].tolist()))
    fila = df.loc[df["country"] == pais, FEATURES].iloc[0]
    nombre = pais
    valores = fila.to_dict()
else:
    st.subheader("Datos del nuevo país")
    st.caption("Ingresa valores en las mismas unidades utilizadas por el dataset original.")
    defaults = df[FEATURES].median().to_dict()
    mins = df[FEATURES].min().to_dict()
    maxs = df[FEATURES].max().to_dict()
    valores = {}
    cols = st.columns(3)
    for i, key in enumerate(FEATURES):
        step = 100.0 if key in ("income", "gdpp") else 0.1
        with cols[i % 3]:
            valores[key] = st.number_input(
                DESCRIPTIONS[key],
                value=float(defaults[key]),
                step=step,
                help=f"Rango observado en el dataset: {mins[key]:,.2f} a {maxs[key]:,.2f}",
            )
    nombre = st.text_input("Nombre del país (opcional)", value="Nuevo país").strip() or "Nuevo país"

x = pd.DataFrame([[valores[f] for f in FEATURES]], columns=FEATURES)
x_scaled = scaler.transform(x)
cluster_raw = int(kmeans.predict(x_scaled)[0])
rango = raw_a_rango[cluster_raw]
grupo = LABELS[rango]
punto_pca = pca.transform(x_scaled)[0]

st.divider()
resultado, datos = st.columns([1, 2])
with resultado:
    st.subheader("Resultado del modelo")
    st.metric("País evaluado", nombre)
    st.metric("Clasificación", grupo)
    if grupo == "Necesidad alta":
        st.error("Prioridad potencial alta: su perfil es similar al grupo con mayores condiciones de vulnerabilidad del conjunto analizado.")
    elif grupo == "Intermedio":
        st.warning("Prioridad intermedia: presenta un perfil de transición entre el grupo de mayor necesidad y el desarrollado.")
    else:
        st.success("Perfil desarrollado dentro del conjunto analizado: menor prioridad relativa según este modelo.")
    st.write(f"**PC1:** {punto_pca[0]:.3f} | **PC2:** {punto_pca[1]:.3f}")

with datos:
    st.subheader("Variables utilizadas")
    tabla = pd.DataFrame({
        "Variable": FEATURES,
        "Descripción": [DESCRIPTIONS[f] for f in FEATURES],
        "Valor": [valores[f] for f in FEATURES],
    })
    st.dataframe(tabla, hide_index=True, use_container_width=True)

st.subheader("Visualizaciones")
tab1, tab2, tab3 = st.tabs(["PCA 2D", "PIB vs. mortalidad infantil", "Perfil de grupos"])

with tab1:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r, etiqueta in enumerate(LABELS):
        mask = rangos == r
        ax.scatter(coords_pca[mask, 0], coords_pca[mask, 1], label=etiqueta, alpha=0.65)
    ax.scatter(punto_pca[0], punto_pca[1], marker="*", s=260, edgecolors="black", linewidths=1.2, label=nombre)
    ax.set_xlabel("Componente principal 1 (PC1)")
    ax.set_ylabel("Componente principal 2 (PC2)")
    ax.set_title("Distribución de países en el espacio PCA")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"PC1 y PC2 explican conjuntamente {pca.explained_variance_ratio_.sum()*100:.1f}% de la varianza estandarizada.")

with tab2:
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for r, etiqueta in enumerate(LABELS):
        mask = rangos == r
        ax2.scatter(df.loc[mask, "gdpp"], df.loc[mask, "child_mort"], label=etiqueta, alpha=0.65)
    ax2.scatter(valores["gdpp"], valores["child_mort"], marker="*", s=260, edgecolors="black", linewidths=1.2, label=nombre)
    ax2.set_xlabel("PIB per cápita (gdpp)")
    ax2.set_ylabel("Mortalidad infantil (child_mort)")
    ax2.set_title("PIB per cápita vs. mortalidad infantil")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

with tab3:
    perfiles = df_app.groupby("grupo")[FEATURES].median().reindex(LABELS)
    st.dataframe(perfiles.round(2), use_container_width=True)
    st.caption("Se utiliza la mediana para disminuir la influencia de valores extremos.")

st.divider()
st.subheader("Países del mismo grupo")
coinciden = (df_app[df_app["grupo"] == grupo][["country", "child_mort", "life_expec", "income", "gdpp"]]
             .sort_values("child_mort", ascending=False))
st.dataframe(coinciden, hide_index=True, use_container_width=True)

st.caption("Proyecto académico — Sistemas Inteligentes — Universidad Nacional de Cajamarca.")
