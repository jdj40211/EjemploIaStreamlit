import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image

st.set_page_config(page_title="Ejemplo IA con Streamlit", layout="wide")

st.title("Ejemplo IA — Streamlit")
st.markdown("Sube un CSV con tus datos para entrenar un clasificador simple (RandomForest). También puedes subir imágenes para visualizarlas.")

# Sección: CSV -> DataFrame -> Entrenamiento rápido
st.header("Entrenar un modelo simple desde CSV")
uploaded_file = st.file_uploader("Sube un CSV", type=["csv"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa del dataset")
    st.write(df.head())

    # Selección de columna objetivo
    target = st.selectbox("Selecciona la columna objetivo (target)", options=df.columns)
    features = [c for c in df.columns if c != target]

    if len(features) == 0:
        st.warning("El dataset no tiene columnas predictoras aparte del target.")
    else:
        selected_features = st.multiselect("Selecciona las columnas predictoras", options=features, default=features[:min(5, len(features))])

        if st.button("Entrenar modelo"):
            # Preparar datos (el código asume que las columnas seleccionadas son numéricas)
            X = df[selected_features].copy()
            y = df[target].copy()

            # Intentar convertir a numérico cuando sea posible
            X = X.apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(y, errors="coerce")

            # Eliminar filas con NaN
            data = pd.concat([X, y], axis=1).dropna()
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
                st.write(cm)

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
st.markdown("Notas: este es un ejemplo didáctico. Para producción, añade manejo de errores, preprocesado más robusto y persistencia de modelos.")
