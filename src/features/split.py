import os
from posixpath import split
import random
from shutil import copyfile
import shutil
import argparse

CLASSES = [
    "0/",
    "1/",
    "2/",
    "3/",
    "4/",
    "5/",
    "6/",
    "7/",
    "8/",
    "9/",
    "A/",
    "B/",
    "C/",
    "D/",
    "E/",
    "F/",
    "G/",
    "H/",
    "I/",
    "J/",
    "K/",
    "L/",
    "M/",
    "N/",
    "R/",
    "S/",
    "T/",
    "U/",
    "W/",
    "Y/",
    "Z/",
    "chevron/",
]


def delete_images_in_dir(split_path):
    for this_class in os.listdir(split_path):
        for filename in os.listdir(split_path + this_class):
            file_path = os.path.join(split_path + this_class, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))


def split_data(
    source_path, training_path, testing_path, split_size, max_class_size=None
):
    for this_class in CLASSES:
        files = []
        for filename in os.listdir(source_path + this_class):
            file = source_path + this_class + filename
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + " is zero length, so ignoring.")

        shuffled_set = random.sample(files, len(files))
        if max_class_size != None:
            if len(shuffled_set) > max_class_size:
                shuffled_set = shuffled_set[0:max_class_size]
        testing_length = int(len(shuffled_set) * split_size)
        training_length = int(len(shuffled_set) - testing_length)
        training_set = shuffled_set[0:training_length]
        testing_set = shuffled_set[-testing_length:]

        os.makedirs(training_path, exist_ok=True)
        for c in CLASSES:
            os.makedirs(training_path + c, exist_ok=True)

        os.makedirs(testing_path, exist_ok=True)
        for c in CLASSES:
            os.makedirs(testing_path + c, exist_ok=True)

        for filename in training_set:
            this_file = source_path + this_class + filename
            destination = training_path + this_class + filename
            copyfile(this_file, destination)

        for filename in testing_set:
            this_file = source_path + this_class + filename
            destination = testing_path + this_class + filename
            copyfile(this_file, destination)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="data/labeled/")
    parser.add_argument("--training_path", type=str, default="data/train/")
    parser.add_argument("--testing_path", type=str, default="data/test/")
    parser.add_argument("--split_size", type=float, default=0.1)
    parser.add_argument("--max_class_size", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_examples = {}
    all_class_dir = os.listdir(args.source_path)
    for all_class in all_class_dir:
        all_examples[all_class] = len(os.listdir(args.source_path + all_class))
    total = 0
    for value in all_examples.values():
        total += value
    print(
        "%%%% Data SET Directory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    print(all_examples)
    print(total)
    if os.path.exists(args.training_path):
        delete_images_in_dir(args.training_path)
    if os.path.exists(args.testing_path):
        delete_images_in_dir(args.testing_path)

    split_data(
        source_path=args.source_path,
        training_path=args.training_path,
        testing_path=args.testing_path,
        split_size=args.split_size,
        max_class_size=args.max_class_size,
    )
