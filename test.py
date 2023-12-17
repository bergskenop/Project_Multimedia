import cv2
from tkinter import filedialog
import numpy as np
import re
import os
import math

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
    cnt = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # cv2.imshow('thresh', cnt)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contour = np.squeeze(cont)
    points = list(zip(contour[:, 0], contour[:, 1]))

    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]
    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]

    sh = [np.abs(min_y - max_y), np.abs(min_x - max_x)]
    print(sh)

    detect_image = np.zeros_like(image)
    detect_image = cv2.cvtColor(detect_image, cv2.COLOR_BGR2GRAY)
    lines = []
    parameter_te_veranderen = 1
    treshold = 150
    minLength = 150
    maxLineGap = 25
    break_outer = False
    y_n = []
    x_n = []
    for i in range(50, 0, -2):
        for j in range(50, 0, -2):
            for k in range(25, 0, -2):
                lines = cv2.HoughLinesP(cnt, 1, np.pi / 180, threshold=i, minLineLength=j, maxLineGap=k)
                if lines is not None and len(lines) >= 4:
                    lines = sorted(lines, key=lambda line: line[0][1], reverse=True)[:4]

                    angles = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        rico = (y2 - y1) / (x2 - x1)
                        angle = math.degrees(math.atan(rico))
                        cv2.line(detect_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
                        if y1 + 10 > y2 > y1 - 10:
                            y_n.append(y1)
                        if x1 + 10 > x2 > x1 - 10:
                            x_n.append(x1)

                    if len(x_n) == 2 and len(y_n) == 2:
                        break_outer = True
                        break  # This will break out of the inner loop
                    else:
                        y_n = []
                        x_n = []
            if break_outer:
                break
        if break_outer:
            break

    # print(x_n, y_n)

    corners = [(x_n[0], y_n[0]), (x_n[1], y_n[0]), (x_n[0], y_n[1]), (x_n[1], y_n[1])]

    # cv2.imshow('detect_image', detect_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for corner in corners:
        cv2.circle(image, corner, 3, (0, 255, 0), -1)

    # intersect_image = np.zeros_like(image)
    # intersect_image = cv2.cvtColor(intersect_image, cv2.COLOR_BGR2GRAY)
    # lines = cv2.HoughLines(detect_image, 1, np.pi / 180, threshold=40)
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(intersect_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # filtered_points_array = filter_points(cont_squeeze, 15)
    # for c in filtered_points_array:
    #     cv2.circle(intersect_image, c, 1, (255, 255, 255), -1)


    # RANSAC PROBEREN !!!!!!!

    # cv2.imshow('intersect_image', intersect_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Display the result
if image.shape[0] > 800 and image.shape[1] > 700:
    image = cv2.resize(image, (int(image.shape[0] / 1.5), int(image.shape[1] / 1.5)))
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()








# code voor in scrambled to rotate dat met houghlineP werkt zodat extra parameters kunnen meegegeven worden i.p.v. houghline gewoon



# for i in range(0, 1):
#             for p, piece in enumerate(temp_pieces):
#                 piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
#                 ret, piece_thresh = cv2.threshold(piece_gray, 0, 254, 0)
#                 kernel = np.ones((3, 3), np.uint8)
#                 dilate = cv2.dilate(piece_thresh, kernel, iterations=1)
#                 erosion = cv2.erode(piece_thresh, kernel, iterations=1)
#                 cnt = cv2.bitwise_xor(erosion, dilate, mask=None)
#                 thres_n = 80
#                 lines = cv2.HoughLines(cnt, 1, np.pi / 180, threshold=thres_n)
#                 _, piece_thresh = cv2.threshold(piece_gray, 0, 254, 0)
#                 # kernel = np.ones((3, 3), np.uint8)
#                 # dilate = cv2.dilate(piece_thresh, kernel, iterations=1)
#                 # erosion = cv2.erode(piece_thresh, kernel, iterations=1)
#                 # cnt = cv2.bitwise_xor(erosion, dilate, mask=None)
#
#                 contour_img = np.zeros_like(self.image)
#                 cv2.drawContours(contour_img, contours, p, (255, 255, 255), thickness=cv2.FILLED)
#                 gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
#
#                 # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#                 cnt = cv2.Canny(gray, 50, 150, apertureSize=3)
#                 # cv2.imshow('test', cnt)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#
#                 # print(piece.shape)
#                 thres_n = 25
#                 min_length = 25
#                 maxLineGap = 10
#                 lines = cv2.HoughLinesP(cnt, 1, np.pi / 180, threshold=thres_n, minLineLength=min_length, maxLineGap=maxLineGap)
#                 angles = []
#                 while lines is None and thres_n >= 20:
#                     thres_n -= 5
#                     lines = cv2.HoughLines(cnt, 1, np.pi / 180, threshold=thres_n)
#                 print(f'threshold value: {thres_n}')
#                     min_length -= 5
#                     lines = cv2.HoughLinesP(cnt, 1, np.pi / 180, threshold=thres_n, minLineLength=min_length, maxLineGap=maxLineGap)
#                     print("test")
#                 # print(f'threshold value: {thres_n}')
#                 if lines is not None:
#                     for line in lines:
#                         rho, theta = line[0]
#                         angle = np.degrees(theta)
#                         x1, y1, x2, y2 = line[0]
#                         rico = (y2 - y1) / (x2 - x1)
#                         angle = math.degrees(math.atan(rico))
#                         if 0 <= angle <= 90:
#                             angles.append(angle)
#                     if len(angles) > 1:
#                         # Compute the median angle
#                         median_angle = np.median(angles)
#                         angle_to_rotate = np.mean(angles)
#                         print(angle_to_rotate)
#
#                         # Calculate the rotation angle
#                         rotation_angle = (90.0 - median_angle)
#                         print(rotation_angle)
#                         rotation_angle = (90.0 - angle_to_rotate)
#                         # print(rotation_angle)
#
#                         piece = imutils.rotate_bound(piece, rotation_angle)
#                     temp_pieces[p] = piece
#
#
# @@ -175,6 +196,7 @@ class Puzzle:
#         else:
#             cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', img)
#         cv2.waitKey(delay)
#         cv2.destroyAllWindows()





# METHODE VOOR HET ZOEKEN VAN 2X2 PUZZELS

# def identify_and_place_corners(pieces, puzzle_dim):
#     height_puzzle_piece = pieces[0].get_height()
#     width_puzzle_piece = pieces[0].get_width()
#     rows, columns, _ = puzzle_dim
#     solved_image = np.zeros([height_puzzle_piece * rows, width_puzzle_piece * columns, 3], dtype=np.uint8)
#     for piece in pieces:
#         min_x, min_y, max_x, max_y = 0, 0, 0, 0
#         # Allign cornerpieces
#         piece_img = piece.get_piece()
#         if piece.get_edges()[0].get_type() == 'straight' and piece.get_edges()[3].get_type() == 'straight':
#             # Top left
#             min_x, min_y = 0, 0
#             max_x = piece_img.shape[0]
#             max_y = piece_img.shape[1]
#
#         elif piece.get_edges()[0].get_type() == 'straight' and piece.get_edges()[1].get_type() == 'straight':
#             # Bottom left
#             min_x = (height_puzzle_piece * rows) - piece_img.shape[0]
#             max_x = height_puzzle_piece * columns
#             min_y = 0
#             max_y = piece_img.shape[1]
#         elif piece.get_edges()[1].get_type() == 'straight' and piece.get_edges()[2].get_type() == 'straight':
#             # Bottom right
#             min_x = (height_puzzle_piece * rows) - piece_img.shape[0]
#             max_x = height_puzzle_piece * rows
#             min_y = (width_puzzle_piece * columns) - piece_img.shape[1]
#             max_y = width_puzzle_piece * columns
#         elif piece.get_edges()[2].get_type() == 'straight' and piece.get_edges()[3].get_type() == 'straight':
#             # Top right
#             min_x = 0
#             max_x = piece_img.shape[0]
#             min_y = (width_puzzle_piece * columns) - piece_img.shape[1]
#             max_y = width_puzzle_piece * columns
#         if max_x != 0:
#             # solved_image[min_x:max_x, min_y:max_y, :] = piece_img
#             temp_image = np.zeros_like(solved_image)
#             temp_image[min_x:max_x, min_y:max_y, :] = piece_img
#             solved_image = cv2.bitwise_or(solved_image, temp_image, mask=None)
#             # solved_image=solved_image
#     cv2.imshow('solved_image', solved_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




