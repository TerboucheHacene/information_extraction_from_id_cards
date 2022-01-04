from typing import List
import imutils
import cv2
import numpy as np
from typing import List


def get_skew_angle(cvImage: np.array) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    # gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(newImage, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Find all contours
    contours, hierarchy = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    # allContourAngles = [cv2.minAreaRect(c)[-1] for c in contours]
    # angle = sum(allContourAngles) / len(allContourAngles)
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = 90 - angle
        angle = -1 * angle
    return -1.0 * angle


def rotate_image(cvImage: np.array, angle: float) -> np.array:
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(
        newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return newImage


# Deskew image
def deskew(cvImage: np.array) -> np.array:
    angle = get_skew_angle(cvImage)
    return rotate_image(cvImage, -1.0 * angle)


def separate_lines(imageB: np.array, imageG: np.array) -> List[np.array]:
    M, N = imageB.shape
    lines = []
    for i in range(0, M):
        if np.all(imageB[i, :] == 0):
            lines.append(i)
    l = []
    cmp = 0
    while cmp < len(lines) - 1:
        D = lines[cmp + 1] - lines[cmp]
        if D > 5:
            l.append(lines[cmp])
            l.append(lines[cmp + 1])
        cmp += 1
    l1 = imageG[l[0] : l[1] + 3, :]
    l2 = imageG[l[2] - 3 : l[3] + 3, :]
    l3 = imageG[l[4] - 3 : l[5] + 3, :]
    images = [l1, l2, l3]
    l1 = imageB[l[0] : l[1] + 3, :]
    l2 = imageB[l[2] - 3 : l[3] + 3, :]
    l3 = imageB[l[4] - 3 : l[5] + 3, :]
    images += [l1, l2, l3]
    return images


def separate_characters(imageB: np.array, imageG: np.array) -> List[List]:
    M, N = imageB.shape
    lines = [0]
    for i in range(0, N):
        if np.all(imageB[:, i] == 0):
            lines.append(i)
    l = []
    cmp = 0
    while cmp < len(lines) - 1:
        D = lines[cmp + 1] - lines[cmp]
        if D > 1:
            l.append(lines[cmp])
            l.append(lines[cmp + 1])
        cmp += 1
    if len(l) < 60:
        l += [N, N]
    if len(l) > 60:
        l = l[:60]
    characters = []
    characters.append(imageG[:, 0 : l[1] + 5])
    cmp = 2
    while cmp < len(l) - 3:
        characters.append(imageG[:, l[cmp] - 3 : l[cmp + 1] + 3])
        cmp += 2
    characters.append(imageG[:, l[-3] :])
    return characters


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
