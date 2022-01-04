import cv2
import imutils
import numpy as np
from utils_old import get_characters_from_image
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List
from pathlib import Path


def _process(image_name: str, input_path: Path, output_path: Path) -> None:
    image_path = cv2.imread(input_path + image_name)
    try:
        total = get_characters_from_image(image_path)
        for i in range(3):
            for j in range(30):
                c = total[i][j]
                dim = (40, 60)
                c = cv2.resize(c, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(
                    output_path + image_name + "_" + str(i) + "_" + str(j) + ".png",
                    c,
                )
    except BaseException as err:
        print(f"Fail: Unexpected {err=}, {type(err)=}")
        print("Can't extract characters from image {}".format(image_name))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i,", type=str, default="data_sample/all/")
    parser.add_argument(
        "--output_path", "-o", type=str, default="data_sample/processed/"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    process = partial(_process, input_path=args.input_path, output_path=args.output_path)
    images = os.listdir(args.input_path)
    images = [
        image
        for image in images
        if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg")
    ]

    with Pool(processes=cpu_count()) as pool:
        pool.map(process, images)
