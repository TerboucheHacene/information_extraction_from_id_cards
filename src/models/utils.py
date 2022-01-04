import tensorflow as tf
import math
import os
import numpy as np
from typing import List, Dict

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


def get_loaders(train_path, validation_path, batch_size) -> List[tf.data.Dataset]:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        shuffle=True,
        seed=123,
        image_size=(60, 40),
        batch_size=batch_size,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        shuffle=False,
        seed=123,
        image_size=(60, 40),
        batch_size=batch_size,
    )

    return train_ds, validation_ds


def get_test_loader(test_path, batch_size) -> tf.data.Dataset:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="grayscale",
        shuffle=False,
        seed=123,
        image_size=(60, 40),
        batch_size=batch_size,
    )
    return test_ds


def get_class_weight(source_path, mu=0.15) -> Dict:
    all_examples = []
    for this_class in CLASSES:
        all_examples += [len(os.listdir(source_path + this_class))]
    labels_dict = dict(enumerate(all_examples))

    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def learning_rate_search_space(lower=1e-4, upper=1e-1, size=5):
    a = np.log10(lower)
    b = np.log10(upper)
    r = (b - a) * np.random.rand(size) + a
    lr = np.power(10, r)
    lr = np.sort(lr)
    return lr


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, name="cm", num_classes=2, normalized=False, **kwargs):
        """initializes attributes of the class"""
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.normalized = normalized
        # Initialize Required variables
        self.c_matrix = tf.Variable(
            initial_value=tf.zeros_initializer()(
                shape=(num_classes, num_classes), dtype=tf.int32
            )
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calulcate confusion matrix.
        y_pred_index = tf.math.argmax(y_pred, axis=1)
        y_true_index = tf.math.argmax(y_true, axis=1)
        conf_matrix = tf.math.confusion_matrix(
            y_true_index, y_pred_index, num_classes=self.num_classes
        )
        self.c_matrix.assign_add(conf_matrix)

    def result(self):
        """Computes and returns the metric value tensor."""
        if self.normalized:
            c = self.c_matrix / tf.reduce_sum(self.c_matrix, axis=0)
            return c
        else:
            return self.c_matrix

    def reset_state(self):
        """Resets all of the metric state variables."""
        # The state of the metric will be reset at the start of each epoch.
        self.c_matrix.assign(0)
