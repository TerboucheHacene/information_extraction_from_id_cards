import sys

sys.path.append("src/app/")
# sys.path.append("src/streamlit/")
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import ExtractInformationFromImage
from schemas import Settings


@st.experimental_singleton
def load_artifacts(_settings):
    inference_system = ExtractInformationFromImage()
    inference_system.load_model(_settings.model_checkpoint)
    print("Artifacts loaded !!")
    return inference_system


settings = Settings()
inference_system = load_artifacts(_settings=settings)
image_input = st.file_uploader(
    "ID Card Image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

col1, col2 = st.columns(2)

if image_input is not None:
    bytes_data = image_input.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_np = np.array(img_np)

    with col1:
        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        st.pyplot(fig)

    with col2:
        information = inference_system.predict(img_np)
        for k, v in information.dict().items():
            st.metric(k, v)
