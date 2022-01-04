import tensorflow as tf
import tensorflow_addons as tfa
from models import Classifier, InferenceClassifier, get_classifier_model
import argparse
from utils import CLASSES


def load_model(args):
    test_sample = tf.random.normal(shape=(1, 60, 40, 1))
    # model = Classifier(num_classes=len(CLASSES))
    model = get_classifier_model(num_classes=len(CLASSES))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-3),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
        ],
    )
    _ = model.predict(test_sample)
    model.load_weights(args.model_checkpoint)
    return model


def save_models(args, model):
    print("Start saving ...")
    tf.saved_model.save(model, args.model_checkpoint + "model/")
    model.save(args.model_checkpoint + "model.h5", save_format="h5")
    test_sample = tf.random.normal(shape=(1, 60, 40, 1))
    infer = InferenceClassifier(classifier=model)
    _ = infer.predict(test_sample)
    tf.saved_model.save(infer, args.model_checkpoint + "serving/")
    # infer.save(args.model_checkpoint + "serving/")
    print("done !!!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="models/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args)
    save_models(args, model)
