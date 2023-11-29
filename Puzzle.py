import cv2
import numpy as np
from segmentation import *

class Puzzle:
    def __init__(self, image_path, type, rows, columns):
        self.contours = None
        self.image = cv2.imread(image_path)
        self.type = type
        self.rows = rows
        self.columns = columns
        self.size = rows * columns

    def set_contours(self):
        self.contours = process_puzzle(self)

    def draw_contours(self):
        img_contours = self.image
        for i in range(0, self.size):
            cv2.drawContours(img_contours, self.contours, i, (0, 255, 0), 3)

    def show(self,  image=None):
        cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}',self.image)
        cv2.waitKey(0)
