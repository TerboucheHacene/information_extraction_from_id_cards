import cv2
import imutils
import numpy as np
from utils import separate_lines, separate_characters, deskew
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List
from pathlib import Path


def get_characters_from_image(image: np.array) -> List[List]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    image = imutils.resize(image, height=600)
    gray = cv2.GaussianBlur(image, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p :] = 0
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])
        if ar > 4 and crWidth > 0.5:
            pX = int((x + w) * 0.05)
            pY = int((y + h) * 0.05)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            roi = image[y : y + h, x : x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
    M, N = roi.shape
    roi[0:1, :] = 255
    roi[M - 1 : M, :] = 255
    roi[:, 0:1] = 255
    roi[:, N - 1 : N] = 255
    roi = deskew(roi)
    (thresh, im_bw1) = cv2.threshold(
        roi, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    M, N = roi.shape
    s = 10
    im_bw1 = im_bw1[s : M - s, s : N - s]
    roi = roi[s : M - s, s : N - s]
    images = separate_lines(im_bw1, roi)
    sub_images = []
    for i in range(0, 3):
        characters = separate_characters(images[i + 3], images[i])
        sub_images.append(characters)
    return sub_images


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
    parser.add_argument("--input_path", type=str, default="data/raw/")
    parser.add_argument("--output_path", type=str, default="data/processed/")
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
