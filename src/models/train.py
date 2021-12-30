import tensorflow as tf
from models import Classifier


def train():
    model = Classifier(num_classes=32)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=8,
        epochs=200,
        verbose=1,
        callbacks=[callbacks],
    )
