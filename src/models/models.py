import tensorflow as tf
from utils import CLASSES


class Classifier(tf.keras.Model):
    def __init__(self, num_classes, conv_layers=[16, 32, 64]) -> None:
        super().__init__()
        self.conv_layers = []
        for conv in conv_layers:
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


class InferenceClassifier(tf.keras.Model):
    def __init__(self, classifier: tf.keras.Model) -> None:
        super().__init__()
        self.classifier = classifier
        self.index_to_class = tf.constant(value=CLASSES, dtype=tf.string)

    def call(self, x):
        y = self.classifier(x)
        class_index = tf.math.argmax(y, axis=1)
        return tf.gather(self.index_to_class, class_index)
