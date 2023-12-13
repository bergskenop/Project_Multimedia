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

ret, thresh = cv2.threshold(gray, 0, 254, 0)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
contours = contours2[:rijen*kolommen]

contour = np.squeeze(contours[0])
list_contours = list(zip(contour[:, 0], contour[:, 1]))

min_x = min(list_contours, key=lambda x: x[0])[0]
max_x = max(list_contours, key=lambda x: x[0])[0]
min_y = min(list_contours, key=lambda x: x[1])[1]
max_y = max(list_contours, key=lambda x: x[1])[1]

part_image = image[min_y:max_y, min_x:max_x, :]
gray = cv2.cvtColor(part_image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the square is the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Find the minimum area rectangle that encloses the contour
rect = cv2.minAreaRect(contours[0])

# Get the rotation angle from the rectangle
angle = rect[2]

# Rotate the image
rows, cols = part_image.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), flags=cv2.INTER_NEAREST)

# Display the result
cv2.imshow('Rotated Square', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
