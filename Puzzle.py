from PuzzlePiece import *
import re
import cv2
import numpy as np
from helper import *


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
        self.contour_draw_fully = None  # Bevat alle punten van een rand
        self.image_path = image_path  # Self expl
        self.image = cv2.imread(image_path)  # Self expl
        self.type = 1  # Type puzzel; 1: shuffled; 2: scrambled; 3: rotated
        self.rows = 1  # Self expl
        self.columns = 1  # Self expl
        self.size = 1  # Hoeveelheid puzzelstukken rows*columns
        self.width_puzzle_piece = None  # Breedte van puzzelstukken
        self.height_puzzle_piece = None  # Hoogte van puzzelstukken
        self.solved_image = None  # Uiteindelijk resultaat komt hier terecht

    def initialise_puzzle(self):
        self.set_puzzle_parameters()  # Parameterbepaling uit filename
        self.set_contour_draw()  # Contour detectie van puzzelstukken
        self.set_puzzle_pieces()  # Individuele puzzelstukken declareren: elk eigen contour en hoekpunten
        self.set_correct_puzzlepiece_size()  # Grootte van puzzelstukken bepalen (uniforme verdeling)

    def set_puzzle_parameters(self):
        type_puzzle = 1
        if re.search('.+_scrambled_.+', self.image_path):
            type_puzzle = 2
        elif re.search('.+_rotated_.+', self.image_path):
            type_puzzle = 3
        self.type = type_puzzle
        scale = re.compile("[0-9][x][0-9]").findall(self.image_path)
        rijen = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
        self.rows = rijen
        kolommen = int(str(re.compile("[0-9]$").findall(scale[0])[0]))
        self.columns = kolommen
        self.size = self.rows * self.columns

    def set_contour_draw(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 254, 0)
        contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        self.contour_draw_fully = contours2[:self.rows * self.columns]

    def set_puzzle_pieces(self, comment=False):
        for contour in self.contour_draw_fully:
            contour = np.squeeze(contour)
            teller = 0
            number_found = 0
            correct_x = []
            correct_y = []
            while teller < contour.shape[0] and number_found < 4:
                value_to_count_x = contour[teller][0]
                value_to_count_y = contour[teller][1]
                if (len(correct_y) < 2 and value_to_count_y not in correct_y
                        and np.count_nonzero(contour[:, 1] == value_to_count_y) > 35):
                    correct_y.append(value_to_count_y)
                    number_found += 1
                if (len(correct_x) < 2 and value_to_count_x not in correct_x
                        and np.count_nonzero(contour[:, 0] == value_to_count_x) > 35):
                    correct_x.append(value_to_count_x)
                    number_found += 1
                teller += 1
            # Hoeken toevoegen van linksboven en zo tegen de klok in
            corners = [(correct_x[0], correct_y[0]), (correct_x[0], correct_y[1]), (correct_x[1], correct_y[1]),
                       (correct_x[1], correct_y[0])]
            list_contours = list(zip(contour[:, 0], contour[:, 1]))

            puzzle_piece = PuzzlePiece(list_contours, corners)
            w = abs(corners[1][0] - corners[2][0])
            h = abs(corners[0][1] - corners[1][1])
            puzzle_piece.set_edges(w, h, self.image.copy())
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
            # puzzle_piece.show_puzzlepiece()  # show seperate images for each piece
            # puzzle_piece.print_puzzlepiece()  # information about individual puzzlepiece

    def set_correct_puzzlepiece_size(self):
        self.height_puzzle_piece = abs(self.puzzle_pieces[0].corners[0][1] - self.puzzle_pieces[0].corners[1][1])
        self.width_puzzle_piece = abs(self.puzzle_pieces[0].corners[1][0] - self.puzzle_pieces[0].corners[2][0])

    def type_based_matching(self):
        # Shuffled 2x2 solver
        self.show(identify_and_place_corners(self.puzzle_pieces, (self.height_puzzle_piece, self.width_puzzle_piece), (self.rows, self.columns, 3)))

    def show(self, img=None):
        if img is None:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', self.image)
        else:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', img)
        cv2.waitKey(0)

    def draw_contours(self):
        img_contours = np.zeros_like(self.image)
        cv2.drawContours(img_contours, self.contour_draw_fully, -1, (0, 255, 0), 1)
        self.show(img_contours)

    def draw_corners(self):
        img_corners = self.image.copy()
        for piece in self.puzzle_pieces:
            for corner in piece.corners:
                cv2.circle(img_corners, corner, 3, (0, 255, 255), -1)
        self.show(img_corners)
