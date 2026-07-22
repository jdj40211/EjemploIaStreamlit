import streamlit as st

st.set_page_config(page_title="Ejemplo IA Streamlit", layout="wide")

st.title("🤖 Ejemplo de IA con Streamlit")

st.write("Bienvenido a esta aplicación de demostración con Streamlit")

# Sidebar
st.sidebar.header("Configuración")
nombre = st.sidebar.text_input("Tu nombre:", "Usuario")

# Contenido principal
st.header(f"Hola, {nombre}! 👋")

st.markdown("""
Esta es una aplicación de ejemplo construida con Streamlit y Python.

**Características:**
- 🚀 Rápida y fácil de desarrollar
- 📊 Ideal para proyectos de IA y datos
- 🎨 Interfaz moderna y responsive
""")

# Ejemplo interactivo
st.subheader("Prueba interactiva")
numero = st.slider("Elige un número:", 1, 100, 50)
st.write(f"Has elegido: {numero}")

# Footer
st.divider()
st.caption("Hecho con ❤️ usando Streamlit")
