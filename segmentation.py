import cv2 as cv
import numpy as np


def segment_image(image_path):
    jigsaw_image = cv.imread(image_path)
    cv.imshow("jigsaw", jigsaw_image)
    cv.waitKey(0)

segment_image("data/Jigsaw_rotated/jigsaw_rotated_2x2_00.png")