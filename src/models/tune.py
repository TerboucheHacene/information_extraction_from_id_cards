import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
import argparse
from models import Classifier
from utils import get_loaders, get_class_weight, CLASSES
import numpy as np


def train_test_model(args, hparams) -> float:
    # Get model and compile
    model = Classifier(num_classes=len(CLASSES))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=hparams[HP_LEARNING_RATE]),
        metrics=[
            tfa.metrics.F1Score(
                num_classes=len(CLASSES), average="macro", name="f1_score"
            )
        ],
    )

    # Get data loaders
    train_ds, validation_ds = get_loaders(
        train_path=args.train_path,
        validation_path=args.validation_path,
        batch_size=hparams[HP_BATCH_SIZE],
    )
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=5,
        verbose=1,
    )
    callbacks = [early_stopping]
    class_weight = get_class_weight(source_path=args.train_path)

    # Fit the model
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        class_weight=class_weight,
        epochs=args.epochs,
        verbose=1,
        callbacks=[callbacks],
    )
    _, f1_score = model.evaluate(validation_ds)
    return f1_score


def run(run_dir, hparams, args):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        f1_score = train_test_model(args, hparams)
        tf.summary.scalar(METRIC_F1Score, f1_score, step=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train/")
    parser.add_argument("--validation_path", type=str, default="data/validation/")
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    learning_rate_search_space = np.linspace(5e-4, 5e-3, 5).tolist()
    HP_LEARNING_RATE = hp.HParam(
        "learning_rate", hp.Discrete(learning_rate_search_space)
    )
    HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([128]))

    METRIC_F1Score = "f1_score"
    with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[HP_LEARNING_RATE, HP_BATCH_SIZE],
            metrics=[hp.Metric(METRIC_F1Score, display_name="F1_Score")],
        )

    session_num = 0
    for batch_size in HP_BATCH_SIZE.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
            hparams = {
                HP_BATCH_SIZE: batch_size,
                HP_LEARNING_RATE: learning_rate,
            }
            run_name = "run-%d" % session_num
            print("--- Starting trial: %s" % run_name)
            print({h.name: hparams[h] for h in hparams})
            run("logs/hparam_tuning/" + run_name, hparams, args)
            session_num += 1
