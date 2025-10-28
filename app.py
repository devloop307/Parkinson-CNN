from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Inicializamos la app Flask
app = Flask(__name__)

# Intentamos cargar el modelo
try:
    model = load_model("modelo.h5", compile=False)
    print("✅ Modelo cargado correctamente sin compilar")
except Exception as e:
    print("⚠️ Error al cargar el modelo:", e)
    model = None

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta de predicción
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ No se pudo cargar el modelo. Verifica el archivo y las versiones."

    if 'file' not in request.files:
        return "⚠️ No se subió ningún archivo"

    file = request.files['file']
    if file.filename == '':
        return "⚠️ No se seleccionó ningún archivo"

    # Guardamos la imagen subida
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Procesamos la imagen para el modelo
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizamos la predicción
    try:
        result = model.predict(img_array)
        prediction = "🧠 Tiene Parkinson" if result[0][0] > 0.5 else "✅ No tiene Parkinson"
    except Exception as e:
        prediction = f"⚠️ Error durante la predicción: {str(e)}"

    # Mostramos el resultado en la página
    return render_template('index.html', prediction=prediction, img_path=filepath)

# Punto de entrada
if __name__ == '__main__':
    app.run(debug=True)
