import sys


sys.path.append("src/models/")
import cv2
import tensorflow as tf
from typing import Optional
import numpy as np
from .schemas import PersonalInformation, Settings
from predict import model_fn, predict_fn

from fastapi import FastAPI, Request, Depends, status, File, UploadFile
from typing import Optional

app = FastAPI(
    title="Personal Information Extraction from ID Cars",
    description="Extract name, birthdate, ID and others informations from a picture of an ID card",
    version="0.0.1",
)

settings = Settings()


class ExtractInformationFromImage:
    model: Optional[tf.keras.Model]

    def load_model(self, settings):
        # self.model = tf.keras.models.load_model("models/model.h5")
        model, loaded = model_fn(settings.model_checkpoint)
        self.loaded = loaded
        self.model = model

    async def predict(self, image: UploadFile = File(...)) -> PersonalInformation:
        if not self.model:
            raise RuntimeError
        data = np.fromfile(image.file, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        informations = predict_fn(model=self.model, input_image=image)
        return PersonalInformation(**informations)


inference_model = ExtractInformationFromImage()


@app.on_event("startup")
def load_artifacts():
    inference_model.load_model(settings)
    print("Ready for inference!")


@app.get("/", tags=["General"], status_code=status.HTTP_200_OK)
def _index(request: Request):
    """Health check."""
    response = {
        "data": "Everything is working as exepected",
    }
    return response


@app.post("/predict", tags=["Prediction"], status_code=status.HTTP_200_OK)
async def _predict(
    output: PersonalInformation = Depends(inference_model.predict),
) -> PersonalInformation:
    """Predict if a review is positive, negative or neutral"""
    # Predict
    return output
