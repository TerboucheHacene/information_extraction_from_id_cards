import sys

sys.path.append("src/models/")
sys.path.append("src/app/")
import cv2
import tensorflow as tf
from typing import Optional
import numpy as np
from schemas import PersonalInformation, Settings
from predict import model_fn, predict_fn


class ExtractInformationFromImage:
    model: Optional[tf.keras.Model]

    def load_model(self, model_checkpoint):
        # self.model = tf.keras.models.load_model("models/model.h5")
        model, loaded = model_fn(model_checkpoint)
        self.loaded = loaded
        self.model = model

    def predict(self, input_image: np.ndarray) -> PersonalInformation:
        if not self.model:
            raise RuntimeError
        informations = predict_fn(model=self.model, input_image=input_image)
        return PersonalInformation(**informations)
