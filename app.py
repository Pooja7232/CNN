import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

st.set_page_config(
    page_title="Rock Paper Scissors CNN",
    page_icon="✊",
    layout="wide"
)

st.title("✊ Rock Paper Scissors Classification")

st.markdown(
"""
CNN Image Classification Dashboard
"""
)

model = tf.keras.models.load_model(
    "models/rps_cnn.keras"
)

classes = [
    "Paper",
    "Rock",
    "Scissors"
]

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded)

    st.image(
        image,
        width=300
    )

    img = image.resize((150,150))

    img = np.array(img)/255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(img)

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    st.success(
        f"Prediction: {classes[predicted_class]}"
    )

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.subheader(
        "Prediction Probabilities"
    )

    for i, cls in enumerate(classes):

        st.progress(
            float(prediction[0][i])
        )

        st.write(
            f"{cls}: {prediction[0][i]*100:.2f}%"
        )