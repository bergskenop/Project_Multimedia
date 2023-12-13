import cv2
import matplotlib.pyplot as plt
import numpy as np


def identify_and_place_corners(pieces, piece_dim, puzzle_dim):
    height_puzzle_piece, width_puzzle_piece = piece_dim
    rows, columns, _ = puzzle_dim
    solved_image = np.zeros([height_puzzle_piece * rows, width_puzzle_piece * columns, 3], dtype=np.uint8)
    for piece in pieces:
        min_x, min_y, max_x, max_y = 0, 0, 0, 0
        # Allign cornerpieces
        piece_img = piece.get_piece()
        if piece.get_edges()[0].get_type() == 'straight' and piece.get_edges()[3].get_type() == 'straight':
            # Top left
            min_x, min_y = 0, 0
            max_x = piece_img.shape[0]
            max_y = piece_img.shape[1]

        elif piece.get_edges()[0].get_type() == 'straight' and piece.get_edges()[1].get_type() == 'straight':
            # Bottom left
            min_x = (height_puzzle_piece * rows) - piece_img.shape[0]
            max_x = height_puzzle_piece * columns
            min_y = 0
            max_y = piece_img.shape[1]
        elif piece.get_edges()[1].get_type() == 'straight' and piece.get_edges()[2].get_type() == 'straight':
            # Bottom right
            min_x = (height_puzzle_piece * rows) - piece_img.shape[0]
            max_x = height_puzzle_piece * rows
            min_y = (width_puzzle_piece * columns) - piece_img.shape[1]
            max_y = width_puzzle_piece * columns
        elif piece.get_edges()[2].get_type() == 'straight' and piece.get_edges()[3].get_type() == 'straight':
            # Top right
            min_x = 0
            max_x = piece_img.shape[0]
            min_y = (width_puzzle_piece * columns) - piece_img.shape[1]
            max_y = width_puzzle_piece * columns
        if max_x != 0:
            # solved_image[min_x:max_x, min_y:max_y, :] = piece_img
            temp_image = np.zeros_like(solved_image)
            temp_image[min_x:max_x, min_y:max_y, :] = piece_img
            solved_image = cv2.bitwise_or(solved_image, temp_image, mask=None)
            # solved_image=solved_image
    return solved_image




def match(pieces, puzzle_dim):
    rows, columns, depth = puzzle_dim  # kunnen we niet weten bij scrambled puzzels
    # Begin puzzelstuk zoeken door het eerste puzzelstuk met 2 rechte lijnen te vinden en dit te draaien tot het
    # hoekpunt linksboven is zodat w esteeds van daaruit vertrekken bij het matchen van puzzelstukken
    i = 0
    corner_found = False
    while not corner_found:
        aantal_rechte_lijnen = 0
        for edge in pieces[i].get_edges():
            # kijken of er twee rechte lijnen in het puzzelstuk zijn
            if edge.get_type() == 'straight':
                aantal_rechte_lijnen += 1
        if aantal_rechte_lijnen == 2:
            # draaien tot de twee rechte lijnen links en boven zitten zodat de hoek linksboven is
            while (not pieces[i].get_edges()[0].get_type() == 'straight'
                   or not pieces[i].get_edges()[3].get_type() == 'straight'):
                pieces[i].rotate(90)
            corner_found = True
        else:
            i += 1
    # i heeft de index van het beginpuzzelstuk
    pieces_copy = pieces
    pieces_solved = [pieces[i]]
    pieces_copy.remove(pieces[i])
    # Bereken de grootte van de opgeloste image op basis van het eerste puzzelstuk
    solved_image = np.zeros([pieces[i].get_height() * rows, pieces[i].get_width() * columns, 3], dtype=np.uint8)

    for number in range(rows * columns - 1):
        newLine = False
        if pieces_solved[number].get_edges()[2].get_type().lower() == 'straight':
            type_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_type()
            hist_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_histogram()
            lengte_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_lengte()
            newLine = True
        else:
            type_of_edge_to_match = pieces_solved[number].get_edges()[2].get_type().lower()
            lengte_of_edge_to_match = pieces_solved[number].get_edges()[2].get_lengte()
            hist_of_edge_to_match = pieces_solved[number].get_edges()[2].get_histogram()
        best_piece = None
        best_piece_edge_number = None
        best_match_value = 10000
        for piece in pieces_copy:
            for n, edge in enumerate(piece.get_edges()):
                if ((edge.get_lengte() + 5 > lengte_of_edge_to_match > edge.get_lengte() - 5) and
                        ((edge.get_type() == 'innie' and type_of_edge_to_match == 'outie') or
                         (edge.get_type() == 'outie' and type_of_edge_to_match == 'innie'))):
                    # cv2.imshow('t', piece.get_piece())
                    # cv2.waitKey(0)
                    value = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=1)
                    print(value)
                    if best_match_value == 1000000 or best_match_value > value:
                        best_match_value = value
                        best_piece = piece
                        best_piece_edge_number = n
        # de index van de rand geeft aan hoeveel graden het puzzelstuk gedraaid moet worden.
        best_piece_copy = best_piece
        if not newLine:
            best_piece_copy.rotate(best_piece_edge_number * 90)
            pieces_solved.append(best_piece_copy)
        else:
            best_piece_copy.rotate((3 - best_piece_edge_number) * 90)
            pieces_solved.append(best_piece_copy)
        pieces_copy.remove(best_piece)

    for piece in pieces_solved:
        cv2.imshow("piece", piece.get_piece())
        cv2.waitKey(0)

        # min_x = 0
        # min_y = 0
        # max_x = pieces_solved[0].get_piece_width()
        # max_y = pieces_solved[0].get_piece_height()
        # for n, piece in enumerate(pieces_solved):
        #     piece_img = piece.get_piece()
        #     temp_image = np.zeros_like(solved_image)
        #
        #     temp_image[min_x:max_x, min_y:max_y, :] = piece_img
        #     solved_image = cv2.bitwise_or(solved_image, temp_image, mask=None)
        #
        #     min_x += piece.get_width()
        #     max_x = piece.get_width() * n + piece.get_piece_width()






    def match_histogram(hist_to_compare, hist_array):
        # method: 0 => correlation, 1 => chi-square, 2 => intersection en 3 => Bhattacharyya
        best_match_index = 0
        best_match_value = cv2.compareHist(hist_to_compare, hist_array[0], method=1)
        for n in range(1, len(hist_array)):
            value = cv2.compareHist(hist_to_compare, hist_array[n], method=1)
            if value < best_match_value:
                best_match_index = n
                best_match_value = value
        return best_match_index





        # We kijken naar de rechterrand van het laatst gevonden puzzelstuk en kijken welke puzzelstukken voldoen aan
        # de eisen om een mogelijk juist puzzelstuk te zijn, dat is namelijk dat ze een innie rand moeten hebben als de
        # rechterrand van het laatst gevonden puzzelstuk een outie was en omgekeer
        # en ze moeten ook ongeveer dezelfde lengte hebben.
        # Wanneer de rechterrand van het laatst gevonden puzzelstuk een rechte lijne is zullen de mogelijk
        # juiste puzzelstukken ook een rechte rand van ongeveer die lengte moeten hebben.