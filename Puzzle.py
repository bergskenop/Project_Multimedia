import cv2

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
        self.solved_image_pieces = None
        self.solved_image = None  # Uiteindelijk resultaat komt hier terecht
        self.nummer = None  # Self expl


    def initialise_puzzle(self):
        self.set_puzzle_parameters()  # Parameterbepaling uit filename
        if self.type == 2:
            self.scrambled2rotated()
        self.set_contour()  # Contour detectie van puzzelstukken
        self.set_puzzle_pieces()
        print("Alle hoeken gedetecteerd")


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
        nummer = re.compile("[0-9].png").findall(self.image_path)
        self.nummer = int(str(nummer[0][0]))


    def set_contour(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 255, 0)
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

            contour = np.squeeze(contour)
            list_contours = list(zip(contour[:, 0], contour[:, 1]))

            min_x = min(list_contours, key=lambda x: x[0])[0]
            max_x = max(list_contours, key=lambda x: x[0])[0]
            min_y = min(list_contours, key=lambda x: x[1])[1]
            max_y = max(list_contours, key=lambda x: x[1])[1]
            piece_w = np.abs(min_x - max_x)
            piece_h = np.abs(min_y - max_y)

            break_outer = False
            y_n = []
            x_n = []

            if self.type == 2:
                if piece_w < 150 or piece_h < 150:
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    cnt = cv2.Canny(blur, 50, 150, apertureSize=5)
                else:
                    cnt = cv2.Canny(gray, 50, 150, apertureSize=5)

                for i in [10, 30, 50, 80, 100]:
                    for j in range(200, 0, -2):
                        for k in range(60, 0, -2):
                            lines = cv2.HoughLinesP(cnt, 1, np.pi / 180, threshold=i, minLineLength=j, maxLineGap=k)
                            if lines is not None and len(lines) >= 4:
                                lines = sorted(lines, key=lambda line: line[0][1], reverse=True)[:4]

                                for line in lines:
                                    x1, y1, x2, y2 = line[0]
                                    if y1 + 10 > y2 > y1 - 10 and (
                                            len(y_n) == 0 or np.abs(y_n[0] - y1) > int(piece_h * 0.6)):
                                        y_n.append(y1)
                                    if x1 + 10 > x2 > x1 - 10 and (
                                            len(x_n) == 0 or np.abs(x_n[0] - x1) > int(piece_w * 0.6)):
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

            else:
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                cnt = cv2.Canny(blurred, 50, 150, apertureSize=3)

                begin = 100
                if piece_w < 250 or piece_h < 250:
                    begin = 75
                if piece_w < 200 or piece_h < 200:
                    begin = 60
                if piece_w < 130:
                    begin = 50
                if piece_w < 85:
                    begin = 30
                for i in range(begin, 0, -5):
                    for j in range(begin, 0, -4):
                        for k in range(25, 0, -4):
                            lines = cv2.HoughLinesP(cnt, 1, np.pi / 180, threshold=i, minLineLength=j, maxLineGap=k)
                            if lines is not None and len(lines) >= 4:
                                lines = sorted(lines, key=lambda line: line[0][1], reverse=True)[:4]

                                for line in lines:
                                    x1, y1, x2, y2 = line[0]
                                    if y1 + 10 > y2 > y1 - 10 and (
                                            len(y_n) == 0 or np.abs(y_n[0] - y1) > int(piece_h * 0.6)):
                                        y_n.append(y1)
                                    if x1 + 10 > x2 > x1 - 10 and (
                                            len(x_n) == 0 or np.abs(x_n[0] - x1) > int(piece_w * 0.6)):
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

            temp_corners = [(x_n[0], y_n[0]), (x_n[1], y_n[0]), (x_n[0], y_n[1]), (x_n[1], y_n[1])]

            corners_in_correct_order = []
            # Robuustere manier voor het vinden van de volgorde van de hoeken, dit werkt ook voor scrambled
            sorted_x = sorted(temp_corners, key=lambda x: x[0])
            sorted_y = sorted(temp_corners, key=lambda x: x[1])
            corners_in_correct_order.append(sorted(sorted_y[:2], key=lambda x: x[0])[0])
            corners_in_correct_order.append(sorted(sorted_x[:2], key=lambda x: x[1])[1])
            corners_in_correct_order.append(sorted(sorted_y[2:], key=lambda x: x[0])[1])
            corners_in_correct_order.append(sorted(sorted_y[:2], key=lambda x: x[0])[1])

            puzzle_piece = PuzzlePiece(list_contours)
            puzzle_piece.set_edges_and_corners(self.image.copy(), corners_in_correct_order, self.size, self.type)

            height_puzzle_piece = abs(corners_in_correct_order[0][1] - corners_in_correct_order[1][1])
            width_puzzle_piece = abs(corners_in_correct_order[1][0] - corners_in_correct_order[2][0])

            puzzle_piece.set_width_and_height(width_puzzle_piece, height_puzzle_piece)
            self.puzzle_pieces.append(puzzle_piece)

            puzzle_piece.set_piece(self.image[min_y:max_y, min_x:max_x, :])
            puzzle_piece.set_piece_width_and_height(piece_w, piece_h)
            # puzzle_piece.show_puzzlepiece()  # show seperate images for each piece
            # puzzle_piece.print_puzzlepiece()  # information about individual puzzlepiece


    def scrambled2rotated(self):
        # We gebruiken 2 enlarge methodes en nemen een bitwise and van de 2 om het beste resultaat te bekomen,
        # dit doen we omdat de contours niet meer juist gedecteerd werden na het draaien avn de image
        if self.size >= 9:
            afb1 = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=2)
            afb1_gray = cv2.cvtColor(afb1, cv2.COLOR_BGR2GRAY)
            afb2 = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=0)
            afb2_gray = cv2.cvtColor(afb2, cv2.COLOR_BGR2GRAY)
            _, thresh1 = cv2.threshold(afb1_gray, 0, 255, 0)
            _, thresh2 = cv2.threshold(afb2_gray, 0, 255, 0)
            mask = cv2.bitwise_and(thresh1, thresh2, mask=None)
            mask = np.stack([mask, mask, mask], axis=2)
            self.image = cv2.bitwise_and(afb1, mask, mask=None)

        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 0, 255, 0)
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
            angle = round(90 - np.degrees(theta), 2)
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


    def match(self):
        # Als de logica een error geeft plaatsen we de stukken in een andere volgorde om dan hopelijk een goed
        # resultaat te bekomen, we doen dit wel maar een beperkt aantal keer anders zou het in een
        # oneindige while loop kunnen geraken
        # Nu zal het programma blijven lopen nadat we uit de while zijn, dit is niet de bedoeling
        teller = 0
        isGelukt = False
        while not isGelukt and teller < 20:
            try:
                self.solved_image = match(self.puzzle_pieces, (self.rows, self.columns, 3), self.nummer, self.type)
                self.show(self.solved_image)
                isGelukt = True
            except TypeError as e:
                random.shuffle(self.puzzle_pieces)
                teller += 1
        if teller == 20:
            raise Exception("Teveel geprobeerd maar geen oplossing gevonden!")


    def show(self, img=None, delay=0):
        show_image = img
        if img is None:
            show_image = self.image

        new_height = 500
        new_width = int(new_height / show_image.shape[0] * show_image.shape[1])
        show_image = cv2.resize(show_image, (new_width, new_height))

        type_stuk_string = "SHUFFLED"
        if self.type == 2:
            type_stuk_string = "SCRAMBLED"
        elif self.type == 3:
            type_stuk_string = "ROTATED"
        cv2.destroyAllWindows()
        cv2.imshow(f'PUZZLE {self.rows}x{self.columns}_0{self.nummer} {type_stuk_string}', show_image)
        cv2.waitKey(1000)


    def draw_contours(self):
        img_contours = np.zeros_like(self.image)
        cv2.drawContours(img_contours, self.contours, -1, (0, 255, 0), 3)
        self.show(img_contours)


    def draw_corners(self):
        img_corners = self.image.copy()
        for piece in self.puzzle_pieces:
            for corner in piece.corners:
                cv2.circle(img_corners, corner, 3, (0, 255, 255), -1)
            self.show(img_corners)
