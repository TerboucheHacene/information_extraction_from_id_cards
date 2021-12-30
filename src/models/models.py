import tensorflow as tf


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

    def call(self, x) -> tf.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten()
        x = self.mlp(x)
        return x
