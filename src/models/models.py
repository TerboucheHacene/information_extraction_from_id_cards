import tensorflow as tf
from utils import CLASSES


class Classifier(tf.keras.Model):
    def __init__(self, num_classes, conv_dims=[16, 32, 64]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv_dims = conv_dims
        self.conv_layers = []
        for conv in conv_dims:
            conv_block = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters=conv, kernel_size=(3, 3), activation="relu"
                    ),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                ]
            )
            self.conv_layers.append(conv_block)
        self.mlp = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        self.scale = tf.keras.layers.Rescaling(1.0 / 255)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x) -> tf.Tensor:
        x = self.scale(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x


def get_classifier_model(num_classes, conv_dims=[16, 32, 64]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(60, 40, 1)))
    model.add(tf.keras.layers.Rescaling(1.0 / 255))
    for conv in conv_dims:
        conv_block = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=conv, kernel_size=(3, 3), activation="relu"
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            ]
        )
        model.add(conv_block)
    model.add(tf.keras.layers.Flatten())
    mlp = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ]
    )
    model.add(mlp)
    return model


class InferenceClassifier(tf.keras.Model):
    def __init__(self, classifier: tf.keras.Model, classes=CLASSES) -> None:
        super().__init__()
        self.classes = classes
        self.classifier = classifier
        self.index_to_class = tf.constant(value=self.classes, dtype=tf.string)

    def call(self, x):
        y = self.classifier(x)
        class_index = tf.math.argmax(y, axis=1)
        return tf.gather(self.index_to_class, class_index)
