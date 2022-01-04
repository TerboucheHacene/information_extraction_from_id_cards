import sys

sys.path.append("src/data/")
import cv2
import tensorflow as tf
import numpy as np
from process import get_characters_from_image
import argparse
from shutil import copyfile
import os

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


def get_unlabeled_dataset(data_path, batch_size):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_path,
        labels=None,
        color_mode="grayscale",
        shuffle=False,
        seed=123,
        image_size=(60, 40),
        batch_size=1,
    )
    path_ds = tf.data.Dataset.from_tensor_slices(test_ds.file_paths)
    ds = tf.data.Dataset.zip((test_ds, path_ds))
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_class_from_prediction(prediction, CLASSES):
    classes = np.array(CLASSES)
    ind = np.argmax(prediction, axis=1)
    return classes[ind]


def model_fn(model_checkpoint):
    # self.model = tf.keras.models.load_model("models/model.h5")
    loaded = tf.saved_model.load(model_checkpoint)
    model = loaded.signatures["serving_default"]
    return model, loaded


def copy_image_to_destination(source_paths, destination_path, classes):
    for (source_path, this_class) in zip(source_paths, classes):
        source_path = source_path.decode("UTF-8")
        file_name = "new" + source_path[22:]
        destination = os.path.join(destination_path, this_class)
        destination = os.path.join(destination, file_name)
        copyfile(source_path, destination)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="models/model/")
    parser.add_argument("--data_path", type=str, default="data_sample/processed/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--labeled_data", type=str, default="data_sample/labeled/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, loaded = model_fn(model_checkpoint=args.model_checkpoint)
    ds = get_unlabeled_dataset(data_path=args.data_path, batch_size=args.batch_size)
    for c in CLASSES:
        os.makedirs(args.labeled_data + c, exist_ok=True)
    for batch in ds:
        model_input = tf.squeeze(batch[0], axis=1)
        source_paths = batch[1].numpy().tolist()
        predictions = model(model_input)["sequential_4"].numpy()
        classes = get_class_from_prediction(predictions, CLASSES)
        copy_image_to_destination(
            source_paths=source_paths,
            destination_path=args.labeled_data,
            classes=classes,
        )
