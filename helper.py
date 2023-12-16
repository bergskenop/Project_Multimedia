import cv2
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


# 5x5_01 en 5x5_03 en 5x5_06 zullen nooit werken omdat hun hoeken slecht gedetecteerd worden,
# verlopig werken 5x5_00 niet bij shuffled en 4x4_00, 5x5_00 en 5x5_08 niet bij rotated !!!
def match(pieces, puzzle_dim):
    rows, columns, depth = puzzle_dim  # rows en columns kan nog omgekeerd staan, dit controleren we later

    # Begin puzzelstuk zoeken door het eerste puzzelstuk met 2 rechte lijnen te vinden en dit te draaien tot het
    # hoekpunt linksboven is zodat we steeds van daaruit vertrekken bij het matchen van puzzelstukken
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
    pieces_copy = pieces.copy()
    pieces_solved = [pieces[i]]
    pieces_copy.remove(pieces[i])

    print(f"rows = {rows}, columns = {columns}")
    grootste_dim = max(rows, columns)
    kleinste_dim = min(rows, columns)
    rij = 0

    for number in range(rows * columns - 1):
        kolom = (number + 1) % columns
        newLine = False
        print(f"Edge 0 is van type: {pieces_solved[number].get_edges()[0].get_type()}, en lengte: {pieces_solved[number].get_edges()[0].get_lengte()}")
        print(f"Edge 1 is van type: {pieces_solved[number].get_edges()[1].get_type()}, en lengte: {pieces_solved[number].get_edges()[1].get_lengte()}")
        print(f"Edge 2 is van type: {pieces_solved[number].get_edges()[2].get_type()}, en lengte: {pieces_solved[number].get_edges()[2].get_lengte()}")
        print(f"Edge 3 is van type: {pieces_solved[number].get_edges()[3].get_type()}, en lengte: {pieces_solved[number].get_edges()[3].get_lengte()}")
        # Geef kolommen en rijen de gepaste waarde,
        # dit kon in het begin omgewisseld zijn naargelang de plaatsing van het eerste puzzelstuk
        if grootste_dim != kleinste_dim and number == 1:
            if (pieces_solved[number].get_edges()[3].get_type() == 'straight'
                    and pieces_solved[number].get_edges()[2].get_type() == 'straight'):
                columns = kleinste_dim
                rows = grootste_dim
                kolom = (number + 1) % columns
                print(f"rows = {rows}, columns = {columns}")
            else:
                columns = grootste_dim
                rows = kleinste_dim
                print(f"rows = {rows}, columns = {columns}")
        # Als we op het einde van een rij zijn, kijk dan naar de onderste rand van het eerste puzzelstuk van de rij
        # i.p.v. de rechtse rand van het vorige puzzelstuk
        if kolom == 0:
            type_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_type()
            hist_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_histogram()
            lengte_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_lengte()
            newLine = True
            rij += 1
            print(newLine)
        else:
            type_of_edge_to_match = pieces_solved[number].get_edges()[2].get_type().lower()
            lengte_of_edge_to_match = pieces_solved[number].get_edges()[2].get_lengte()
            hist_of_edge_to_match = pieces_solved[number].get_edges()[2].get_histogram()

        # cv2.imshow(f'piece {number}', pieces_solved[number].get_piece())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(f"lengte_of_edge_to_match => {lengte_of_edge_to_match}")
        print(f"type_of_edge_to_match => {type_of_edge_to_match}")
        best_piece = None
        best_piece_edge_number = None
        best_match_value1 = 100000
        # best_match_value2 = 100000
        # best_match_value3 = 0
        # best_match_value4 = 0
        lijst_van_index_foute_pieces_en_randen = []

        logicIsOk = False
        while not logicIsOk:
            for k, piece in enumerate(pieces_copy):
                for n, edge in enumerate(piece.get_edges()):
                    if ((edge.get_lengte() + 5 > lengte_of_edge_to_match > edge.get_lengte() - 5) and
                            ((edge.get_type() == 'innie' and type_of_edge_to_match == 'outie') or
                             (edge.get_type() == 'outie' and type_of_edge_to_match == 'innie')) and
                            (len(lijst_van_index_foute_pieces_en_randen) == 0 or not
                            any((t[0] == k and t[1] == n) for t in lijst_van_index_foute_pieces_en_randen))):
                        # method: 0 => correlation, 1 => chi-square, 2 => intersection en 3 => Bhattacharyya
                        value1 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=3)
                        # value2 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=1)
                        # value3 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=0)
                        # value4 = cv2.compareHist(hist_of_edge_to_match, edge.get_histogram(), method=2)
                        # print(f"value => {value1}")
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
            if newLine:
                rotate_angle = 360 - (3 - best_piece_edge_number) * 90
                print(f"Rotate =================> {rotate_angle}")
                best_piece.rotate(rotate_angle)

            else:
                rotate_angle = best_piece_edge_number * 90
                print(f"Rotate =================> {rotate_angle}")
                best_piece.rotate(rotate_angle)

            # Als het beste stuk gevonden is kijken of het mogelijk is a.d.h.v. logica dat randen aan de buitenkant
            # moeten liggen, als dit niet zo is alles opnieuw onderzoeken, maar de foutieve rand niet meer bekijken
            # Later aanpassen naar np.any i.p.v. for loop
            heeft_rechte_rand = False
            for edge in best_piece.get_edges():
                if edge.get_type() == 'straight':
                    heeft_rechte_rand = True

            # Geeft een fout bij 2x3 omdat we hier in het begin nog niet weten wat de exacte aantal rows en columns is
            # doordat het eerste puzzelstuk gedraaid kan zijn, later eventueel op een manier robuuster maken?
            # Is niet echt nodig want 2x3 doet hij steeds goed en heeft hiervoor dus geen extra logica nodig,
            # alles wat hier in deze if staat van logica is eigenlijk pas van toepassing vanaf 3x3
            if (rows == columns and ((rij == 0 and not best_piece.get_edges()[3].get_type() == 'straight') or
                                     (rij == (rows - 1) and not best_piece.get_edges()[1].get_type() == 'straight') or
                                     (kolom == 0 and not best_piece.get_edges()[0].get_type() == 'straight') or
                                     (kolom == (columns - 1) and not best_piece.get_edges()[2].get_type() == 'straight') or
                                     (heeft_rechte_rand and rij != 0 and rij != (rows - 1)
                                      and kolom != 0 and kolom != (columns - 1)) or
                                     (best_piece.get_edges()[3].get_type() == 'straight' and
                                      best_piece.get_edges()[2].get_type() == 'straight' and
                                      (rij != 0 or kolom != (columns - 1))) or
                                     (best_piece.get_edges()[0].get_type() == 'straight' and
                                      best_piece.get_edges()[1].get_type() == 'straight' and
                                      (rij != (rows - 1) or kolom != 0)) or
                                     (best_piece.get_edges()[1].get_type() == 'straight' and
                                      best_piece.get_edges()[2].get_type() == 'straight' and
                                      (rij != (rows - 1) or kolom != (columns - 1))) or
                                     (rij != 0 and pieces_solved[(number+1) - columns].get_edges()[1].get_type() ==
                                      best_piece.get_edges()[3].get_type()))):
                # Terugzetten naar de originele toestand zoals ze in pieces_copy staan
                best_piece.rotate(360 - rotate_angle)
                lijst_van_index_foute_pieces_en_randen.append((pieces_copy.index(best_piece), best_piece_edge_number))
                print(f"logica fout => puzzelstuk: {pieces_copy.index(best_piece)} en rand {best_piece_edge_number}")
                best_piece = None
                best_piece_edge_number = None
                best_match_value1 = 100000
                # best_match_value2 = 100000
                # best_match_value3 = 0
                # best_match_value4 = 0
            else:
                logicIsOk = True

        pieces_solved.append(best_piece)
        pieces_copy.remove(best_piece)
    print(f"Edge 0 is van type: {pieces_solved[len(pieces_solved) - 1].get_edges()[0].get_type()}, en lengte: {pieces_solved[len(pieces_solved) - 1].get_edges()[0].get_lengte()}")
    print(f"Edge 1 is van type: {pieces_solved[len(pieces_solved) - 1].get_edges()[1].get_type()}, en lengte: {pieces_solved[len(pieces_solved) - 1].get_edges()[1].get_lengte()}")
    print(f"Edge 2 is van type: {pieces_solved[len(pieces_solved) - 1].get_edges()[2].get_type()}, en lengte: {pieces_solved[len(pieces_solved) - 1].get_edges()[2].get_lengte()}")
    print(f"Edge 3 is van type: {pieces_solved[len(pieces_solved) - 1].get_edges()[3].get_type()}, en lengte: {pieces_solved[len(pieces_solved) - 1].get_edges()[3].get_lengte()}")
    # cv2.imshow('laatste piece', pieces_solved[len(pieces_solved)-1].get_piece())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for n, piece in enumerate(pieces_solved):
    #     cv2.imshow(f"piece {n+1}", piece.get_piece())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    solved_width = 0
    solved_height = 0
    pieces_solved = np.array(pieces_solved).reshape(rows, columns)
    # print(pieces_solved.shape)
    for row in range(rows):
        for col in range(columns):
            if pieces_solved[row][col].get_piece_height() > solved_height:
                solved_height = pieces_solved[row][col].get_piece_height()
            if pieces_solved[row][col].get_piece_width() > solved_width:
                solved_width = pieces_solved[row][col].get_piece_width()

    solved_image = np.zeros([solved_height * rows, solved_width * columns, 3], dtype=np.uint8)
    min_y = 0
    max_y = 0
    # print(f'rows: {rows}, columns: {columns}')
    for row, row_pieces in enumerate(pieces_solved):
        min_x = 0
        max_x = 0

        for column, piece in enumerate(row_pieces):
            max_x += piece.get_piece_width()
            max_y = min_y + piece.get_piece_height()
            # cv2.imshow('next piece', piece.get_piece())
            # cv2.waitKey(0)
            # print(
            #     f'position: ({row}, {column}) -> {piece.get_height()} by {piece.get_width()} and {piece.get_piece_height()} by {piece.get_piece_width()} ')

            temp_img = np.zeros_like(solved_image)
            temp_img[min_y:max_y, min_x:max_x, :] = piece.get_piece()
            solved_image = cv2.bitwise_or(solved_image, temp_img, mask=None)
            min_x = max_x

        min_y += solved_height
    cv2.imshow('solved_image', solved_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
