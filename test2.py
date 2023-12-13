import cv2
from tkinter import filedialog
import numpy as np
import re
import os

# PYTHON SCRIPT OM GEPASTE PARAMETERS VOOR HET VINDEN VAN DE HOEKEN TE PROBEREN!!


# path = filedialog.askopenfilename(initialdir="*/", title="Select image",
#                                   filetypes=(("Images", "*.png*"), ("all files", "*.*")))
# path = 'data/Jigsaw_shuffled/jigsaw_shuffled_2x3_01.png'

# image = cv2.imread(path)

for subdir, dirs, files in os.walk("data/Jigsaw_scrambled"):
    for file in files:

        path = os.path.join(subdir, file)

        image = cv2.imread(path)

        scale = re.compile("[0-9][x][0-9]").findall(path)
        rijen = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
        kolommen = int(str(re.compile("[0-9]$").findall(scale[0])[0]))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 254, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:rijen*kolommen]
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        thresh = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

        # blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        # _, thresh = cv2.threshold(blur, 0, 254, 0)

        cv2.imshow('Corners', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # harris corner detection
        corners = cv2.goodFeaturesToTrack(thresh, maxCorners=rijen*kolommen*4, qualityLevel=0.2, minDistance=40,
                                          blockSize=8, useHarrisDetector=True, k=0.21)
        corners = np.int32(corners)

        for c in corners:
            x, y = c.ravel()
            image = cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        if image.shape[0] > 700 and image.shape[1] > 700:
            image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Corners', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

