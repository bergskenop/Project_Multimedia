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

    # Dimensie instellen voor oneven puzzelpieces die geen rechthoek zijn (2x3 alles behalve 05 en 07)
    grootste_dim = max(rows, columns)
    kleinste_dim = min(rows, columns)
    if pieces_solved[0].get_width() < pieces_solved[0].get_height():
        columns = grootste_dim
        rows = kleinste_dim
    else:
        columns = kleinste_dim
        rows = grootste_dim
    print(f"rows, columns => {rows, columns}")
    for number in range(rows * columns - 1):
        newLine = False
        print("############")
        print(f"Edge 0 is van type: {pieces_solved[number].get_edges()[0].get_type()}, en lengte: {pieces_solved[number].get_edges()[0].get_lengte()}")
        print(f"Edge 1 is van type: {pieces_solved[number].get_edges()[1].get_type()}, en lengte: {pieces_solved[number].get_edges()[1].get_lengte()}")
        print(f"Edge 2 is van type: {pieces_solved[number].get_edges()[2].get_type()}, en lengte: {pieces_solved[number].get_edges()[2].get_lengte()}")
        print(f"Edge 3 is van type: {pieces_solved[number].get_edges()[3].get_type()}, en lengte: {pieces_solved[number].get_edges()[3].get_lengte()}")
        # Als we het 2de puzzelstuk behandeld hebben bij een 2x3 met vierkante vorm (2x3 alles behalve 05 en 07)
        if (grootste_dim != kleinste_dim and pieces_solved[0].get_width() + 5 >
                pieces_solved[0].get_height() > pieces_solved[0].get_width() - 5 and number == 1):
            if pieces_solved[number].get_edges()[3].get_type() == 'straight' and pieces_solved[number].get_edges()[2].get_type() == 'straight':
                columns = kleinste_dim
                rows = grootste_dim
            else:
                columns = grootste_dim
                rows = kleinste_dim
        if pieces_solved[number].get_edges()[2].get_type().lower() == 'straight':
            type_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_type()
            hist_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_histogram()
            lengte_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_lengte()
            newLine = True
            print(newLine)
        else:
            type_of_edge_to_match = pieces_solved[number].get_edges()[2].get_type().lower()
            lengte_of_edge_to_match = pieces_solved[number].get_edges()[2].get_lengte()
            hist_of_edge_to_match = pieces_solved[number].get_edges()[2].get_histogram()
        cv2.imshow(f'piece {number}', pieces_solved[number].get_piece())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"lengte_of_edge_to_match => {lengte_of_edge_to_match}")
        print(f"type_of_edge_to_match => {type_of_edge_to_match}")
        best_piece = None
        best_piece_edge_number = None
        best_match_value1 = 100000
        # Het toevoegen van meerdere histogram matching methodes heeft geen effect op 2x3_01, hier wert chi-kwadraat wel het slechtste
        # best_match_value2 = 100000
        # best_match_value3 = 0
        # best_match_value4 = 0
        for piece in pieces_copy:
            for n, edge in enumerate(piece.get_edges()):
                if ((edge.get_lengte() + 5 > lengte_of_edge_to_match > edge.get_lengte() - 5) and
                        ((edge.get_type() == 'innie' and type_of_edge_to_match == 'outie') or
                         (edge.get_type() == 'outie' and type_of_edge_to_match == 'innie'))):
                    # method: 0 => correlation, 1 => chi-square, 2 => intersection en 3 => Bhattacharyya
                    value1 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=3)
                    # print(f"value => {value1}")
                    # value2 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=3)
                    # value3 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=0)
                    # value4 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=2)
                    # cv2.imshow(f'mogelijke best piece met rand {n}', piece.get_piece())
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    if best_match_value1 > value1:  # and best_match_value2 > value2 and best_match_value3 < value3 and best_match_value4 < value4:
                        best_match_value1 = value1
                        # best_match_value2 = value2
                        # best_match_value3 = value3
                        # best_match_value4 = value4
                        best_piece = piece
                        best_piece_edge_number = n
        # de index van de rand geeft aan hoeveel graden het puzzelstuk gedraaid moet worden.
        # best_piece_copy = best_piece
        if not newLine:
            print(f"Rotate ==============> {best_piece_edge_number * 90}")
            best_piece.rotate(best_piece_edge_number * 90)
            pieces_solved.append(best_piece)
        else:
            print(f"Rotate ==============> {360-(3 - best_piece_edge_number) * 90}")
            best_piece.rotate(360-(3 - best_piece_edge_number) * 90)
            pieces_solved.append(best_piece)
        pieces_copy.remove(best_piece)
    print("############")
    print(f"Edge 0 is van type: {pieces_solved[len(pieces_solved)-1].get_edges()[0].get_type()}, en lengte: {pieces_solved[len(pieces_solved)-1].get_edges()[0].get_lengte()}")
    print(f"Edge 1 is van type: {pieces_solved[len(pieces_solved)-1].get_edges()[1].get_type()}, en lengte: {pieces_solved[len(pieces_solved)-1].get_edges()[1].get_lengte()}")
    print(f"Edge 2 is van type: {pieces_solved[len(pieces_solved)-1].get_edges()[2].get_type()}, en lengte: {pieces_solved[len(pieces_solved)-1].get_edges()[2].get_lengte()}")
    print(f"Edge 3 is van type: {pieces_solved[len(pieces_solved)-1].get_edges()[3].get_type()}, en lengte: {pieces_solved[len(pieces_solved)-1].get_edges()[3].get_lengte()}")
    cv2.imshow('laatste piece', pieces_solved[len(pieces_solved)-1].get_piece())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # for n, piece in enumerate(pieces_solved):
    #     cv2.imshow(f"piece {n+1}", piece.get_piece())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # maal_y = -1
    # for n, piece in enumerate(pieces_solved):
    #     piece_img = piece.get_piece()
    #     temp_image = np.zeros_like(solved_image)
    #
    #     if (n+1) % 2 == 0:
    #         min_x = piece.get_width()
    #     else:
    #         min_x = 0
    #     if n % 2 == 0:
    #         maal_y += 1
    #     min_y = maal_y * piece.get_height()
    #     max_x = min_x + piece.get_piece_width()
    #     max_y = min_y + piece.get_piece_height()
    #
    #     temp_image[min_y:max_y, min_x:max_x, :] = piece_img
    #     solved_image = cv2.bitwise_or(solved_image, temp_image, mask=None)
    #     # print(solved_image.shape)  # (726, 480, 3)
    #
    #     cv2.imshow("solved_image", solved_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()






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