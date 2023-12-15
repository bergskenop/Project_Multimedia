import cv2
from tkinter import filedialog
import numpy as np
import re
import os

# PYTHON SCRIPT OM GEPASTE PARAMETERS VOOR HET VINDEN VAN DE HOEKEN TE PROBEREN!!


path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                  filetypes=(("Images", "*.png*"), ("all files", "*.*")))
# path = 'data/Jigsaw_shuffled/jigsaw_shuffled_5x5_01.png'

image = cv2.imread(path)

# for subdir, dirs, files in os.walk("data/Jigsaw_shuffled"):
#     for file in files:
#
#         path = os.path.join(subdir, file)

original_image = cv2.imread(path)
print(original_image.shape)

# if original_image.shape[0] < 400 and original_image.shape[1] < 400:
#     new_width = int(original_image.shape[1] * 1.2)
#     new_height = int(original_image.shape[0] * 1.2)
#     image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
# else:
#     image = original_image

scale = re.compile("[0-9][x][0-9]").findall(path)
rijen = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
kolommen = int(str(re.compile("[0-9]$").findall(scale[0])[0]))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 254, 0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = contours[:rijen*kolommen]

for n, cont in enumerate(contours):
    contour_img = np.zeros_like(image)
    cv2.drawContours(contour_img, contours, n, (255, 255, 255), thickness=cv2.FILLED)

    gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(gray, kernel, iterations=1)
    erosion = cv2.erode(gray, kernel, iterations=1)
    cnt = cv2.bitwise_xor(erosion, dilate, mask=None)

    # qual=0.1, minDist=10, blocksize=7, k=0.21 => alles shuffled/rotated behalve 3 puzzels => 5x5 01, 03 en 06
    # Werkt nog niet goed voor scrambled puzzelstukken
    corners = cv2.goodFeaturesToTrack(cnt, maxCorners=4, qualityLevel=0.1, minDistance=10,
                                      blockSize=7, useHarrisDetector=True, k=0.21)

    # cv2.imshow('thresh', cnt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corners = np.int32(corners)

    for c in corners:
        x, y = c.ravel()
        image = cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # if image.shape[0] > 700 and image.shape[1] > 700:
    #     image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

