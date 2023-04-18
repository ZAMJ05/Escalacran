import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)

# Establecer las opciones de estilo de la página
st.set_page_config(
    page_title="Clasificador de escorpiones - Escalacran",
    page_icon=":scorpion:",
    layout="wide",
    initial_sidebar_state="collapsed",
    )


st.title("Clasificador de escorpiones")

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_train.h5')

# Definir las etiquetas de las clases
class_names = ['Especie Diplocentrus', 'Especie Vaejovis']

species_info = {
    "Especie Diplocentrus": "La mayoría de las picaduras de escorpiones Diplocentrus provocan síntomas locales como dolor, enrojecimiento e hinchazón en el área de la picadura. Estos síntomas suelen ser leves y desaparecen por sí solos en unas pocas horas o días. Sin embargo, algunas personas pueden experimentar reacciones más graves, como náuseas, vómitos, dificultad para respirar o incluso reacciones alérgicas.\n\nSi te pica un escorpión Diplocentrus, sigue estos pasos:\n\n- Lava la zona afectada con agua y jabón para eliminar cualquier rastro de veneno y reducir el riesgo de infección.\n- Aplica hielo o una bolsa de hielo envuelta en un paño sobre la picadura para ayudar a aliviar el dolor y la hinchazón.\n- Mantén la zona de la picadura elevada, si es posible, para reducir la hinchazón.\n- Toma un analgésico de venta libre, como paracetamol o ibuprofeno, para aliviar el dolor si es necesario.\n- Si experimentas síntomas más graves, como dificultad para respirar, vómitos, mareos o una reacción alérgica, busca atención médica de inmediato.\n\nAunque las picaduras de escorpiones Diplocentrus suelen ser de baja toxicidad, siempre es mejor prevenir y tomar precauciones cuando se trata de escorpiones. Si no estás seguro de la especie que te ha picado, busca atención médica para garantizar un tratamiento adecuado.",
    "Especie Vaejovis": "En la mayoría de los casos, una picadura de un escorpión Vaejovis provoca síntomas locales como dolor, enrojecimiento e hinchazón en el área de la picadura. Estos síntomas suelen ser leves y desaparecen por sí solos en unas pocas horas o días. Sin embargo, algunas personas pueden experimentar reacciones más graves, como náuseas, vómitos, dificultad para respirar o incluso reacciones alérgicas.\n\nSi te pica un escorpión Vaejovis, sigue estos pasos:\n\n- Lava la zona afectada con agua y jabón para eliminar cualquier rastro de veneno y reducir el riesgo de infección.\n- Aplica hielo o una bolsa de hielo envuelta en un paño sobre la picadura para ayudar a aliviar el dolor y la hinchazón.\n- Mantén la zona de la picadura elevada, si es posible, para reducir la hinchazón.\n- Toma un analgésico de venta libre, como paracetamol o ibuprofeno, para aliviar el dolor si es necesario.\n- Si experimentas síntomas más graves, como dificultad para respirar, vómitos, mareos o una reacción alérgica, busca atención médica de inmediato.\n\nEs importante recordar que aunque las picaduras de Vaejovis suelen ser de baja toxicidad, siempre es mejor prevenir y tomar precauciones cuando se trata de escorpiones.",
}

# Función para preprocesar la imagen
def preprocess_image(image):
    # Cambiar el tamaño de la imagen a 224x224
    image = image.resize((224, 224))
    # Convertir la imagen a un array de numpy
    image_array = np.array(image)
    # Reescalar los valores de los pixeles entre 0 y 1
    image_array = image_array / 255.0
    # Añadir una dimensión adicional para que el modelo pueda procesarla
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Función para predecir la clase de la imagen
def predict(image):
    # Preprocesar la imagen
    image_array = preprocess_image(image)
    # Predecir la clase
    prediction = model.predict(image_array)
    # Obtener el índice de la clase con mayor probabilidad
    predicted_class = np.argmax(prediction)
    # Devolver la etiqueta de la clase
    return class_names[predicted_class]

# Definir la interfaz gráfica
uploaded_file = st.file_uploader("Cargar imagen", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada.', use_column_width=True, width=50)
    st.write("")
    st.write("Clasificando...")
    label = predict(image)
    st.markdown(f'<p style="font-size:24px">La imagen cargada corresponde a un escorpión de la especie: {label}</p>', unsafe_allow_html=True)
    
    # Mostrar información de la especie
    st.write(species_info[label])