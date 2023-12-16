import os

from PuzzlePiece import *
import re
from helper import *
import random


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
        self.set_contour_draw()  # Contour detectie van puzzelstukken
        self.set_puzzle_pieces()

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

    # Is corners vinden door lijnen te trekken met de gevonden angle zoals bij
    # scrambled2rotated en de 4 doorsnedes te nemen geen optie??????
    def set_puzzle_pieces(self, comment=False):
        for n, contour in enumerate(self.contours):
            contour_img = np.zeros_like(self.image)
            cv2.drawContours(contour_img, self.contours, n, (255, 255, 255), thickness=cv2.FILLED)
            gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(gray, kernel, iterations=1)
            erosion = cv2.erode(gray, kernel, iterations=1)
            cnt = cv2.bitwise_xor(erosion, dilate, mask=None)
            # qual=0.1, minDist=10, blocksize=7, k=0.21 => alles shuffled/rotated behalve 3 puzzels => 5x5 01, 03 en 06
            # Werkt nog niet goed voor scrambled puzzelstukken
            corners = cv2.goodFeaturesToTrack(cnt, maxCorners=4, qualityLevel=0.1, minDistance=10,
                                              blockSize=7, useHarrisDetector=True, k=0.21)
            corners = np.int32(corners)
            temp_corners = []
            for c in corners:
                x, y = c.ravel()
                temp_corners.append((x, y))
            corners_in_correct_order = []

            # temp_img = self.image.copy()
            # for corner in temp_corners:
            #     cv2.circle(temp_img, corner, 3, (0, 255, 255), -1)
            # self.show(temp_img)

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
            puzzle_piece.set_piece_width_and_height(np.abs(min_x - max_x), np.abs(min_y - max_y))
            # puzzle_piece.show_puzzlepiece()  # show seperate images for each piece
            # puzzle_piece.print_puzzlepiece()  # information about individual puzzlepiece

    def scrambled2rotated(self):
        for i in range(1):
            img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray, 0, 254, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = contours[:self.size]
            temp_pieces = []
            for contour in contours:
                contour = np.squeeze(contour)
                list_contour = list(zip(contour[:, 0], contour[:, 1]))
                min_x = min(list_contour, key=lambda x: x[0])[0]
                max_x = max(list_contour, key=lambda x: x[0])[0]
                min_y = min(list_contour, key=lambda x: x[1])[1]
                max_y = max(list_contour, key=lambda x: x[1])[1]
                temp_pieces.append(self.image[min_y - 5:max_y + 5, min_x - 5:max_x + 5, :])

            for p, piece in enumerate(temp_pieces):
                piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
                ret, piece_thresh = cv2.threshold(piece_gray, 0, 254, 0)
                edges = cv2.Canny(piece_thresh, 50, 150, apertureSize=3)
                thresh_n = 80
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh_n)
                while lines is None or len(lines) < 4:
                    thresh_n -= 1
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh_n)
                lines = sorted(lines, key=lambda line: line[0][1], reverse=True)
                piece_cpy = piece.copy()
                rho, theta = lines[0][0]
                angle = round(90 - np.degrees(theta),2)
                print(f'interation {i} : {angle}')
                rotate = imutils.rotate_bound(piece_cpy, angle)
                temp_pieces[p] = rotate
            rotated_list = temp_pieces
            self.place_pieces(rotated_list)

    def place_pieces(self, pieces):
        max_height = max(piece.shape[0] for piece in pieces)
        max_width = max(piece.shape[1] for piece in pieces)
        rotated_puzzle = np.zeros([max_height * self.rows, max_width * self.columns, 3], dtype=np.uint8)

        for i in range(self.rows):
            for j in range(self.columns):
                piece = pieces[i * self.columns + j]

                y_offset = i * max_height
                x_offset = j * max_width

                rotated_puzzle[y_offset:y_offset + piece.shape[0], x_offset:x_offset + piece.shape[1]] = piece
        self.image = rotated_puzzle.copy()
        self.type = 3
        # desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        # folder_name = 'output_folder'
        # output_folder = os.path.join(desktop_path, folder_name)
        # os.makedirs(output_folder, exist_ok=True)
        # output_path = os.path.join(output_folder, self.image_path.replace('\\', '-').replace('/', '_'))
        # print(output_path)
        # cv2.imwrite(output_path, self.image)
        # self.show(delay=0)

    def match(self):
        # Als de logica een error geeft plaatsen we de stukken in een andere volgorde om dan hopelijk een goed
        # resultaat te bekomen, we doen dit wel maar een beperkt aantal keer anders zou het in een
        # oneindige while loop kunnen geraken
        # Nu zal het programma blijven lopen nadat we uit de while zijn, dit is niet de bedoeling
        teller = 0
        isGelukt = False
        while not isGelukt and teller < 10:
            try:
                match(self.puzzle_pieces, (self.rows, self.columns, 3))
                isGelukt = True
            except TypeError as e:
                print("ERROR")
                random.shuffle(self.puzzle_pieces)
                teller += 1
                isGelukt=True
        if teller >= 10:
            raise Exception("Your error message here")

    def show(self, img=None, delay=0):
        if img is None:
            cv2.imshow(f'Puzzle {self.rows}x{self.columns} {self.type}', self.image)
        else:
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
