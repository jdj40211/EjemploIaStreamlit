import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image

st.set_page_config(page_title="Ejemplo IA con Streamlit", layout="wide")

st.title("Ejemplo IA — Streamlit")
st.markdown("Sube un CSV para ver análisis cuantitativo, cualitativo y gráficas interactivas. También puedes entrenar un clasificador simple o subir imágenes.")

# Sección: CSV -> análisis + entrenamiento
st.header("Análisis de CSV")
uploaded_file = st.file_uploader("Sube un CSV", type=["csv"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa del dataset")
    st.write(df.head())

    st.subheader("Información general")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", len(df))
    with col2:
        st.metric("Columnas", len(df.columns))
    with col3:
        st.metric("Nulos totales", int(df.isna().sum().sum()))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["Cuantitativo", "Cualitativo", "Gráficas", "Modelo"])

    with tab1:
        st.subheader("Resumen cuantitativo")
        if numeric_cols:
            st.write(df[numeric_cols].describe().T)
        else:
            st.info("No se detectaron columnas numéricas.")

    with tab2:
        st.subheader("Resumen cualitativo")
        if categorical_cols:
            selected_cat = st.selectbox("Selecciona una columna cualitativa", options=categorical_cols)
            freq = df[selected_cat].astype(str).value_counts().reset_index()
            freq.columns = [selected_cat, "frecuencia"]
            st.write(freq)
            fig_bar = px.bar(freq, x=selected_cat, y="frecuencia", title=f"Frecuencia de {selected_cat}")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No se detectaron columnas cualitativas.")

    with tab3:
        st.subheader("Gráficas interactivas")
        if numeric_cols:
            selected_num = st.selectbox("Selecciona una columna numérica", options=numeric_cols, key="num_col")
            fig_hist = px.histogram(df, x=selected_num, nbins=20, title=f"Histograma de {selected_num}")
            st.plotly_chart(fig_hist, use_container_width=True)

            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr(numeric_only=True)
                fig_heat = px.imshow(corr, text_auto=True, title="Matriz de correlación")
                st.plotly_chart(fig_heat, use_container_width=True)

                x_col = st.selectbox("Eje X", options=numeric_cols, index=0, key="x_scatter")
                y_col = st.selectbox("Eje Y", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="y_scatter")
                fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"Dispersión: {x_col} vs {y_col}")
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No hay columnas numéricas para graficar.")

    with tab4:
        st.subheader("Entrenar un modelo simple desde CSV")
        target = st.selectbox("Selecciona la columna objetivo (target)", options=df.columns, key="target")
        features = [c for c in df.columns if c != target]

        if len(features) == 0:
            st.warning("El dataset no tiene columnas predictoras aparte del target.")
        else:
            selected_features = st.multiselect("Selecciona las columnas predictoras", options=features, default=features[:min(5, len(features))])

            if st.button("Entrenar modelo"):
                X = df[selected_features].copy().apply(pd.to_numeric, errors="coerce")
                y = pd.to_numeric(df[target], errors="coerce")
                data = pd.concat([X, y.rename(target)], axis=1).dropna()
                X = data[selected_features]
                y = data[target]

                if len(y) < 10:
                    st.warning("Pocos datos después de limpiar NaN; necesita al menos 10 filas para entrenar con este ejemplo.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)

                    acc = accuracy_score(y_test, preds)
                    cm = confusion_matrix(y_test, preds)

                    st.success(f"Entrenamiento completado — Accuracy: {acc:.3f}")
                    st.subheader("Matriz de confusión")
                    fig_cm = px.imshow(cm, text_auto=True, title="Matriz de confusión")
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.subheader("Predicciones de ejemplo")
                    example = X_test.head(5).copy()
                    example["pred"] = clf.predict(example)
                    st.write(example)

# Sección: subida y visualización de imágenes
st.header("Visualizar imágenes")
img_file = st.file_uploader("Sube una imagen (jpg/png)", type=["png", "jpg", "jpeg"], key="img")
if img_file is not None:
    img = Image.open(img_file)
    st.image(img, caption="Imagen subida", use_column_width=True)

st.markdown("---")
st.markdown("Notas: este ejemplo ahora incluye análisis cuantitativo, cualitativo y gráficas interactivas con Plotly.")
