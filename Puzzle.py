from PuzzlePiece import *
import re
import cv2
import numpy as np
from helper import *
import imutils


# Logica achter klassenverdeling
#   Elke puzzel bevat
#       - Meerdere puzzelstukken
#       - Bestandslocatie
#       - Afbeelding van de puzzel
#       - Vaste dimensies: (rijen; kolommen; aantal en grootte van puzzelstukken)
#       - Opgeloste afbeelding van de puzzel


class Puzzle:
    def __init__(self, image_path):
        self.puzzle_pieces = []  # Bevat een lijst van verschillende puzzelstukken
        self.contours = None  # Bevat alle punten van een rand
        self.image_path = image_path  # Self expl
        self.image = cv2.imread(image_path)  # Self expl
        self.type = 1  # Type puzzel; 1: shuffled; 2: scrambled; 3: rotated
        self.rows = 1  # Self expl
        self.columns = 1  # Self expl
        self.size = 1  # Hoeveelheid puzzelstukken rows*columns
        self.solved_image = None  # Uiteindelijk resultaat komt hier terecht

    def initialise_puzzle(self):
        self.set_puzzle_parameters()  # Parameterbepaling uit filename
        if self.type == 2:
            self.scrambled2rotated()
        # self.set_contour_draw()  # Contour detectie van puzzelstukken
        # self.set_puzzle_pieces()

    def set_puzzle_parameters(self):
        type_puzzle = 1
        if re.search('.+_scrambled_.+', self.image_path):
            type_puzzle = 2
        elif re.search('.+_rotated_.+', self.image_path):
            type_puzzle = 3
        self.type = type_puzzle
        scale = re.compile("[0-9][x][0-9]").findall(self.image_path)
        self.rows = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
        self.columns = int(str(re.compile("[0-9]$").findall(scale[0])[0]))
        self.size = self.rows * self.columns

    def set_contour_draw(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 254, 0)
        contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        self.contours = contours2[:self.size]

    def set_puzzle_pieces(self, comment=False):
        for n, contour in enumerate(self.contours):
            contour_img = np.zeros_like(self.image)
            cv2.drawContours(contour_img, self.contours, n, (255, 255, 255), thickness=cv2.FILLED)
            gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            # qual=0.1, minDist=10, blocksize=7, k=0.21 => alles shuffled/rotated behalve 3 puzzels => 5x5 01, 03 en 06
            # Werkt nog niet goed voor scrambled puzzelstukken

            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(gray, kernel, iterations=1)
            erosion = cv2.erode(gray, kernel, iterations=1)
            cnt = cv2.bitwise_xor(erosion, dilate, mask=None)

            corners = cv2.goodFeaturesToTrack(cnt, maxCorners=4, qualityLevel=0.1, minDistance=10,
                                              blockSize=7, useHarrisDetector=True, k=0.21)
            corners = np.int32(corners)
            temp_corners = []
            for c in corners:
                x, y = c.ravel()
                temp_corners.append((x, y))
            corners_in_correct_order = []
            volgorde = [3, 1, 0, 2]
            for v in volgorde:
                corners_in_correct_order.append(temp_corners[v])
            contour = np.squeeze(contour)
            list_contours = list(zip(contour[:, 0], contour[:, 1]))
            puzzle_piece = PuzzlePiece(list_contours)
            puzzle_piece.set_edges_and_corners(self.image.copy(), corners_in_correct_order)

            height_puzzle_piece = abs(corners_in_correct_order[0][1] - corners_in_correct_order[1][1])
            width_puzzle_piece = abs(corners_in_correct_order[1][0] - corners_in_correct_order[2][0])

            puzzle_piece.set_width_and_height(width_puzzle_piece, height_puzzle_piece)
            self.puzzle_pieces.append(puzzle_piece)

            # Elke puzzlepiece wordt een cutout van de originele afbeelding meegegeven.
            points = puzzle_piece.get_points()
            if comment:
                print(f'X: ({min(points, key=lambda x: x[0])[0]} -> {max(points, key=lambda x: x[0])[0]})')
                print(f'Y: ({min(points, key=lambda x: x[1])[1]} -> {max(points, key=lambda x: x[1])[1]})')

            min_x = min(points, key=lambda x: x[0])[0]
            max_x = max(points, key=lambda x: x[0])[0]
            min_y = min(points, key=lambda x: x[1])[1]
            max_y = max(points, key=lambda x: x[1])[1]

            puzzle_piece.set_piece(self.image[min_y:max_y, min_x:max_x, :])
            puzzle_piece.show_puzzlepiece()  # show seperate images for each piece
            # puzzle_piece.print_puzzlepiece()  # information about individual puzzlepiece

    def scrambled2rotated(self):
        img_copy = self.image.copy()
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 254, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[:self.size]
        # self.show(img_copy, delay=0)
        # for i, contour in enumerate(contours):
        #     cv2.drawContours(img_copy, contours, 16, (255, 0, 255), thickness=cv2.FILLED)
        #     self.show(img_copy, delay=0)
        temp_pieces = []
        for contour in contours:
            contour = np.squeeze(contour)
            list_contour = list(zip(contour[:, 0], contour[:, 1]))
            min_x = min(list_contour, key=lambda x: x[0])[0]
            max_x = max(list_contour, key=lambda x: x[0])[0]
            min_y = min(list_contour, key=lambda x: x[1])[1]
            max_y = max(list_contour, key=lambda x: x[1])[1]
            temp_pieces.append(self.image[min_y-5:max_y+5, min_x-5:max_x+5, :])

        for p,piece in enumerate(temp_pieces):
            piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
            ret, piece_thresh = cv2.threshold(piece_gray, 0, 254, 0)
            edges = cv2.Canny(piece_thresh, 50, 150, apertureSize=3)
            for i in range(0, 90):
                rotate = imutils.rotate_bound(edges, i)
                img_copy = imutils.rotate_bound(piece, i)
                lines = cv2.HoughLines(rotate, 1, np.pi/180, threshold=80)
                lines_hv = 0
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        angle = np.degrees(theta)
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        if 0 == angle  or angle == 90:
                            lines_hv += 1
                            # cv2.line(img_copy, (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))),
                            #              (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))), (0, 0, 255), 2)
                if lines_hv == 4:
                    temp_pieces[p] = imutils.rotate_bound(piece, i)
                    # cv2.imshow('rotation', piece)
                    break
        for piece in temp_pieces:
            self.show(piece, delay=200)
        return 0

    # def type_based_matching(self):
    #     # Shuffled 2x2 solver
    #     self.show(match(self.puzzle_pieces,
    #                                          (self.height_puzzle_piece, self.width_puzzle_piece),
    #                                          (self.rows, self.columns, 3)))

    def show(self, img=None, delay=20):
        if img is None:
            # cv2.namedWindow(f'Puzzle {self.rows}x{self.columns} {self.type}', cv2.WINDOW_AUTOSIZE)
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', self.image)
        else:
            # cv2.namedWindow(f'Puzzle {self.rows}x{self.columns} {self.type}', cv2.WINDOW_KEEPRATIO)
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', img)
        cv2.waitKey(delay)

    def draw_contours(self):
        img_contours = np.zeros_like(self.image)
        cv2.drawContours(img_contours, self.contours, -1, (0, 255, 0), 1)
        self.show(img_contours)

    def draw_corners(self):
        img_corners = self.image.copy()
        for piece in self.puzzle_pieces:
            for corner in piece.corners:
                cv2.circle(img_corners, corner, 3, (0, 255, 255), -1)
        self.show(img_corners)
