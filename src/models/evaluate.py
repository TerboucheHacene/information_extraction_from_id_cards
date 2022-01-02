import tensorflow as tf
import tensorflow_addons as tfa
from models import Classifier, InferenceClassifier
import argparse
from utils import get_test_loader, CLASSES, ConfusionMatrix
import json
from typing import Dict


def evaluate_model(args, model) -> Dict[str, float]:
    test_loader = get_test_loader(test_path=args.test_path, batch_size=args.batch_size)
    results = model.evaluate(test_loader)
    performance = {
        "loss": results[0],
        "f1_score": results[1],
        "cm": results[2].tolist(),
        "accuracy": results[3],
    }
    return performance


def load_model(args) -> tf.keras.Model:
    test_sample = tf.random.normal(shape=(1, 60, 40, 1))
    model = Classifier(num_classes=len(CLASSES))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-3),
        metrics=[
            tfa.metrics.F1Score(
                num_classes=len(CLASSES), average="macro", name="f1_score"
            ),
            ConfusionMatrix(num_classes=len(CLASSES)),
            tf.keras.metrics.CategoricalAccuracy(),
        ],
    )
    _ = model.predict(test_sample)
    model.load_weights(args.model_checkpoint)
    return model


def save_models(args, model) -> None:
    print("Start saving ...")
    # tf.saved_model.save(model, args.model_checkpoint + "serving/")
    model.save(args.model_checkpoint + "model/", include_optimizer=False)

    test_sample = tf.random.normal(shape=(1, 60, 40, 1))
    infer = InferenceClassifier(classifier=model)
    _ = infer.predict(test_sample)
    # tf.saved_model.save(infer, args.model_checkpoint + "infer/")
    infer.save(args.model_checkpoint + "serving/", include_optimizer=False)
    print("done !!!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="data/test/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_checkpoint", type=str, default="models/")
    parser.add_argument("--result_path", type=str, default="reports/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args)

    performance = evaluate_model(args, model)
    with open(args.result_path + "performance.json", "w") as f:
        json.dump(performance, f)
    print(performance)

    save_models(args, model)
