import cv2
import numpy as np
from segmentation import *


class Puzzle:
    def __init__(self, image_path, type, rows, columns):
        self.contours = None
        self.contourCorners = None
        self.image = cv2.imread(image_path)
        self.type = type
        self.rows = rows
        self.columns = columns
        self.size = rows * columns
        self.breedtepuzzelstuk = None
        self.hoogtepuzzelstuk = None

    def set_contours(self):
        self.contours, self.contourCorners, self.breedtepuzzelstuk, self.hoogtepuzzelstuk = process_puzzle(self)

    def draw_contours(self):
        img_contours = self.image.copy()
        for i in range(0, self.size):
            cv2.drawContours(img_contours, self.contours, i, (0, 255, 0), 3)
        self.show(img_contours)

    def draw_corners(self):
        img_corners = self.image.copy()
        for contour in self.contourCorners:
            for corner in contour:
                cv2.circle(img_corners, corner, 2, (0, 255, 255), -1)
        self.show(img_corners)

    def show(self, image=None):
        if image is None:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', self.image)
        else:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', image)
        cv2.waitKey(0)
