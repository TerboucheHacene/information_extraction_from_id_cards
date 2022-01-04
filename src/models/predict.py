import sys

sys.path.append("src/data/")
import cv2
import tensorflow as tf
import numpy as np
from process import get_characters_from_image
import argparse

CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<",
]


def get_used_characters(characters):
    line_01 = [characters[0][i] for i in range(5, 14)]
    line_02 = [characters[1][i] for i in range(0, 15)]
    line_03 = characters[2]
    images = line_01 + line_02 + line_03
    return images


"""
def get_class_from_prediction(prediction, CLASSES):
    classes = np.array(CLASSES)
    ind = np.argmax(prediction, axis=1)
    return classes[ind]
"""


def get_class_from_prediction(predictions, CLASSES):
    mask = np.ones_like(predictions)
    classes = np.array(CLASSES)
    indices_m_and_n = np.where(((classes != "M") & (classes != "F")))[0]
    mask[0:15, 10:] = 0
    mask[16, indices_m_and_n] = 0
    mask[17:24, 10:] = 0
    mask[24:, 0:10] = 0
    classes = np.array(CLASSES)
    constrainted_prediction = np.multiply(predictions, mask)
    ind = np.argmax(constrainted_prediction, axis=1)
    return classes[ind]


def get_personal_information_from_predictions(outputs):
    line0 = outputs[0:9]
    line1 = outputs[9:24]
    line2 = outputs[24:]
    id_number = "".join(line0)
    birthdate = "".join(line1[0:6])
    birthdate = birthdate[0:2] + "-" + birthdate[2:4] + "-" + birthdate[4:]
    sex = line1[7]
    exp_date = "".join(line1[8:14])
    exp_date = exp_date[0:2] + "-" + exp_date[2:4] + "-" + exp_date[4:]
    last_name = ""
    first_name = ""
    i = 0
    prec = False
    actual = False
    sep = False
    while (i < len(line2)) and not (sep):
        c = line2[i]
        prec = actual
        if c != "<":
            last_name += c
            actual = False
        else:
            last_name += " "
            actual = True
        sep = actual and prec
        i += 1
    prec = False
    actual = False
    sep = False
    while (i < len(line2)) and not (sep):
        c = line2[i]
        prec = actual
        if c != "<":
            first_name += c
            actual = False
        else:
            first_name += " "
            actual = True
        sep = actual and prec
        i += 1
    informations = {
        "first_name": first_name,
        "last_name": last_name,
        "birthdate": birthdate,
        "sex": sex,
        "identity_num": id_number,
        "expiration_date": exp_date,
    }
    return informations


def predict_fn(model, input_image):
    all_characters = get_characters_from_image(input_image)
    characters_to_predict = get_used_characters(all_characters)
    model_input = [tf.cast(image, dtype=tf.float32) for image in characters_to_predict]
    model_input = [tf.expand_dims(image, axis=-1) for image in model_input]
    model_input = tf.stack(model_input, axis=0)
    # predictions = self.model.predict(model_input)
    predictions = model(model_input)["sequential_4"].numpy()
    outputs = get_class_from_prediction(predictions, CLASSES)
    informations = get_personal_information_from_predictions(outputs)
    return informations


def model_fn(model_checkpoint):
    # self.model = tf.keras.models.load_model("models/model.h5")
    loaded = tf.saved_model.load(model_checkpoint)
    model = loaded.signatures["serving_default"]
    return model, loaded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="models/model")
    parser.add_argument("--input_image", type=str, default="data/raw/1.jpg")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image = cv2.imread(args.input_image)
    model, loaded = model_fn(model_checkpoint=args.model_checkpoint)
    response = predict_fn(model=model, input_image=image)
    print()
    print()
    for k, v in response.items():
        print(k, v)
