from PuzzlePiece import *
import re
import cv2
import numpy as np


class Puzzle:
    def __init__(self, image_path):
        self.puzzle_pieces = []
        self.contour_draw = None
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.type = 1
        self.rows = 1
        self.columns = 1
        self.size = 1
        self.width_puzzle_piece = None
        self.height_puzzle_piece = None

    def initialise_puzzle(self):
        self.set_puzzle_parameters()
        self.set_contour_draw()
        self.set_puzzle_pieces()
        self.set_correct_puzzlepiece_size()

    def set_puzzle_parameters(self):
        # 1 = shuffled, 2 = scrambled and 3 = rotated
        type_puzzle = 1
        if re.search(".+_scrambled_.+", self.image_path):
            type_puzzle = 2
        elif re.search(".+_rotated_.+", self.image_path):
            type_puzzle = 3
        self.type = type_puzzle
        # Bepaal aantal rijen en kolommen
        scale = re.compile("[0-9][x][0-9]").findall(self.image_path)
        rijen = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
        self.rows = rijen
        kolommen = int(str(re.compile("[0-9]$").findall(scale[0])[0]))
        self.columns = kolommen

    def set_contour_draw(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 254, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        self.contour_draw = contours[:self.rows * self.columns]

    def set_puzzle_pieces(self):
        for contours in self.contour_draw:
            contour = np.squeeze(contours)
            contour = np.vstack([contour, [10, 10]])
            distances = np.linalg.norm(np.diff(contour, axis=0), axis=1)
            # Later min distance robuuster maken
            contour = np.squeeze(np.array([contour[i] for i in np.where(distances > 10)]))

            unique_elements, counts = np.unique(contour[:, 0], return_counts=True)
            elements_to_remove = unique_elements[counts == 1]
            indexes = []
            for i in elements_to_remove:
                indexes.append(np.squeeze(np.where(contour[:, 0] == i)))
            contour = np.delete(contour, indexes, axis=0)

            unique_elements, counts = np.unique(contour[:, 1], return_counts=True)
            elements_to_remove = unique_elements[counts == 1]
            indexes2 = []
            for i in elements_to_remove:
                indexes2.append(np.where(contour[:, 1] == i))
            contour = np.delete(contour, indexes2, axis=0)

            corners = []
            for i in range(4):
                corners.append((contour[i][0], contour[i][1]))
            contours = np.squeeze(contours)
            list_contours = list(zip(contours[:, 0], contours[:, 1]))
            puzzle_piece = PuzzlePiece(list_contours, corners)
            puzzle_piece.set_edges()
            self.puzzle_pieces.append(puzzle_piece)

    def set_correct_puzzlepiece_size(self):
        self.height_puzzle_piece = abs(self.puzzle_pieces[0].corners[0][1] - self.puzzle_pieces[0].corners[1][1])
        self.width_puzzle_piece = abs(self.puzzle_pieces[0].corners[1][0] - self.puzzle_pieces[0].corners[2][0])

    def show(self, img=None):
        if img is None:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', self.image)
        else:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', img)
        cv2.waitKey(0)

    def draw_contours(self):
        img_contours = self.image.copy()
        cv2.drawContours(img_contours, self.contour_draw, -1, (0, 255, 0), 3)
        self.show(img_contours)

    def draw_corners(self):
        img_corners = self.image.copy()
        for piece in self.puzzle_pieces:
            for corner in piece.corners:
                cv2.circle(img_corners, corner, 2, (0, 255, 255), -1)
        self.show(img_corners)

