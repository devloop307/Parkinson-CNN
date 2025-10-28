import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuración de la página
st.set_page_config(
    page_title="Detector de Parkinson",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Título
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Detección de Parkinson</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sube una imagen de trazo para predecir la probabilidad de Parkinson.</p>", unsafe_allow_html=True)
st.markdown("---")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    try:
        modelo = load_model("modelo_parkinson.h5", compile=False)
        st.success("✅ Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"⚠️ Error al cargar el modelo: {e}")
        modelo = None
    return modelo

modelo = cargar_modelo()

# Función de predicción
def predecir_imagen(imagen):
    img = imagen.convert("RGB").resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = modelo.predict(img_array)[0][0]
    return pred

# Subir imagen
imagen_subida = st.file_uploader("Sube una imagen (trazo de espiral u onda)", type=["jpg", "jpeg", "png"])

if imagen_subida is not None:
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="🖼️ Imagen cargada", use_column_width=True)

    if st.button("🔍 Predecir"):
        if modelo is not None:
            probabilidad = predecir_imagen(imagen)
            if probabilidad > 0.5:
                st.error(f"🧠 Probabilidad de Parkinson detectada: {probabilidad*100:.2f}%")
            else:
                st.success(f"✅ Imagen saludable detectada: {(1 - probabilidad)*100:.2f}%")
        else:
            st.error("❌ No se pudo cargar el modelo. Verifica el archivo .h5")

st.markdown("---")
st.markdown("**Nota:** Este resultado es orientativo y no sustituye una evaluación médica profesional.", unsafe_allow_html=True)
