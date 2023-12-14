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

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = contours[:rijen*kolommen]


# def distance(point1, point2):
#     return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
#
# # Function to filter points based on distance threshold
# def filter_points(points, threshold):
#     new_points = [points[0]]  # Include the first point in the filtered list
#
#     for i in range(1, len(points) - 1):
#         prev_point = new_points[-1]
#         next_point = points[i + 1]
#
#         if distance(prev_point, points[i]) >= threshold and distance(points[i], next_point) >= threshold:
#             new_points.append(points[i])
#
#     new_points.append(points[-1])  # Include the last point in the filtered list
#
#     return np.array(new_points)


for n, cont in enumerate(contours):
    contour_img = np.zeros_like(image)
    # cont_squeeze = np.squeeze(cont)
    # for c in cont_squeeze:
    #     cv2.circle(contour_img, c, 1, (255, 255, 255), -1)

    cv2.drawContours(contour_img, contours, n, (255, 255, 255), thickness=cv2.FILLED)
    # cnt = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((3, 3), np.uint8)
    # dilate = cv2.dilate(gray, kernel, iterations=1)
    # erosion = cv2.erode(gray, kernel, iterations=1)
    # cnt = cv2.bitwise_xor(erosion, dilate, mask=None)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cnt = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imshow('thresh', cnt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detect_image = np.zeros_like(image)
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cnt, 1, np.pi / 180, threshold=25, minLineLength=10, maxLineGap=25)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(detect_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.imshow('detect_image', detect_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    intersect_image = np.zeros_like(image)
    intersect_image = cv2.cvtColor(intersect_image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(detect_image, 1, np.pi / 180, threshold=40)
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
        cv2.line(intersect_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # filtered_points_array = filter_points(cont_squeeze, 15)
    # for c in filtered_points_array:
    #     cv2.circle(intersect_image, c, 1, (255, 255, 255), -1)


    # RANSAC PROBEREN !!!!!!!

    cv2.imshow('intersect_image', intersect_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display the result
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
