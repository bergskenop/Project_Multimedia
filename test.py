import cv2
from tkinter import filedialog
import numpy as np
import re
import os

path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                  filetypes=(("Images", "*.png*"), ("all files", "*.*")))
# path = 'data/Jigsaw_shuffled/jigsaw_shuffled_2x3_01.png'

image = cv2.imread(path)

scale = re.compile("[0-9][x][0-9]").findall(path)
rijen = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
kolommen = int(str(re.compile("[0-9]$").findall(scale[0])[0]))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
print(edges.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 254, 0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = contours[:rijen*kolommen]
contour_img = np.zeros_like(image)
cv2.drawContours(contour_img, contours, -1, (255, 255, 255))
thresh = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

# Apply Hough Line Transform
lines = cv2.HoughLines(thresh, 1, np.pi / 180, threshold=100)

# Draw the lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
