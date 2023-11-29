import cv2
import numpy as np
from segmentation import *
from PuzzlePiece import *


class Puzzle:
    def __init__(self, image_path, type, rows, columns):
        self.puzzle_pieces = []
        self.contour_draw = None
        self.image = cv2.imread(image_path)
        self.type = type
        self.rows = rows
        self.columns = columns
        self.size = rows * columns
        self.breedtepuzzelstuk = None
        self.hoogtepuzzelstuk = None

    def set_contours(self):
        contours_punten, contoursCorners = process_puzzle(self)
        self.contour_draw = contours_punten
        for contour_punten, contourCorners in zip(contours_punten, contoursCorners):
            self.puzzle_pieces.append(PuzzlePiece(contour_punten, contourCorners))

    def set_puzzelstuk_dimensies(self):
        self.breedtepuzzelstuk, self.hoogtepuzzelstuk = getCorrectPuzzleSize(self)

    def draw_contours(self):
        img_contours = self.image.copy()
        print(type(self.puzzle_pieces))
        cv2.drawContours(img_contours, self.contour_draw, -1, (0, 255, 0), 3)
        self.show(img_contours)

    def draw_corners(self):
        img_corners = self.image.copy()
        for contour in self.puzzle_pieces:
            for corner in contour.corners:
                cv2.circle(img_corners, corner, 2, (0, 255, 255), -1)
        self.show(img_corners)

    def show(self, image=None):
        if image is None:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', self.image)
        else:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', image)
        cv2.waitKey(0)
