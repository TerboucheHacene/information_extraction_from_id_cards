import tensorflow as tf
from models import Classifier, get_classifier_model
import argparse
from utils import get_loaders, get_class_weight, CLASSES
import tensorflow_addons as tfa


def train(args):
    # Get model and compile
    # model = Classifier(num_classes=len(CLASSES))
    model = get_classifier_model(num_classes=len(CLASSES))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tfa.metrics.F1Score(
                num_classes=len(CLASSES), average="macro", name="f1_score"
            ),
        ],
    )

    # Get data loaders
    train_ds, validation_ds = get_loaders(
        train_path=args.train_path,
        validation_path=args.validation_path,
        batch_size=args.batch_size,
    )
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=5,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint_path,
        monitor="f1_score",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        save_freq="epoch",
    )

    callbacks = [tensorboard_callback, checkpoint, early_stopping]

    # get class weight
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train/")
    parser.add_argument("--validation_path", type=str, default="data/validation/")
    parser.add_argument("--checkpoint_path", type=str, default="models/")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0020145)
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
