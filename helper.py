import cv2
import numpy as np



def match(pieces, puzzle_dim, nummer, type_stuk):
    rows, columns, depth = puzzle_dim  # rows en columns kan nog omgekeerd staan, dit controleren we later

    # Begin puzzelstuk zoeken door het eerste puzzelstuk met 2 rechte lijnen te vinden en dit te draaien tot het
    # hoekpunt linksboven is zodat we steeds van daaruit vertrekken bij het matchen van puzzelstukken
    i = 0
    corner_found = False
    aantal_keer_geprobeerd = 0
    while not corner_found and aantal_keer_geprobeerd < 20:
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
        aantal_keer_geprobeerd += 1

    if aantal_keer_geprobeerd == 20:
        raise Exception("Geen begin hoekpunt gevonden!")
    # i heeft de index van het beginpuzzelstuk
    pieces_copy = pieces.copy()
    pieces_solved = [pieces[i]]
    pieces_copy.remove(pieces[i])

    grootste_dim = max(rows, columns)
    kleinste_dim = min(rows, columns)
    rij = 0

    for number in range(rows * columns - 1):
        kolom = (number + 1) % columns
        newLine = False
        # Geef kolommen en rijen de gepaste waarde,
        # dit kon in het begin omgewisseld zijn naargelang de plaatsing van het eerste puzzelstuk
        if grootste_dim != kleinste_dim and number == 1:
            if (pieces_solved[number].get_edges()[3].get_type() == 'straight'
                    and pieces_solved[number].get_edges()[2].get_type() == 'straight'):
                columns = kleinste_dim
                rows = grootste_dim
                kolom = (number + 1) % columns
            else:
                columns = grootste_dim
                rows = kleinste_dim
        # Als we op het einde van een rij zijn, kijk dan naar de onderste rand van het eerste puzzelstuk van de rij
        # i.p.v. de rechtse rand van het vorige puzzelstuk
        if kolom == 0:
            type_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_type()
            hist_of_edge_to_match_above = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_histogram()
            lengte_of_edge_to_match = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_lengte()
            hist_of_edge_to_match_right = None
            newLine = True
            rij += 1
        else:
            type_of_edge_to_match = pieces_solved[number].get_edges()[2].get_type().lower()
            hist_of_edge_to_match_right = pieces_solved[number].get_edges()[2].get_histogram()
            lengte_of_edge_to_match = pieces_solved[number].get_edges()[2].get_lengte()
            if rij >= 1:
                hist_of_edge_to_match_above = pieces_solved[len(pieces_solved) - columns].get_edges()[1].get_histogram()
            else:
                hist_of_edge_to_match_above = None

        # cv2.imshow(f'piece {number}', pieces_solved[number].get_piece())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        best_piece = None
        best_piece_edge_number = None
        best_match_value = 100000
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
                        if kolom == 0:
                            value_above = cv2.compareHist(hist_of_edge_to_match_above, edge.get_histogram(), method=3)
                            value = value_above
                        elif rij == 0:
                            value_right = cv2.compareHist(hist_of_edge_to_match_right, edge.get_histogram(), method=3)
                            value = value_right
                        else:
                            value_right = cv2.compareHist(hist_of_edge_to_match_right, edge.get_histogram(), method=3)
                            value_above = cv2.compareHist(hist_of_edge_to_match_above,
                                                          piece.get_edges()[n - 1].get_histogram(), method=3)
                            value = np.mean([value_right, value_above])
                        if best_match_value > value:
                            best_match_value = value
                            best_piece = piece
                            best_piece_edge_number = n
            # de index van de rand geeft aan hoeveel graden het puzzelstuk gedraaid moet worden.
            if newLine:
                rotate_angle = 360 - (3 - best_piece_edge_number) * 90
                best_piece.rotate(rotate_angle)

            else:
                rotate_angle = best_piece_edge_number * 90
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
                                     (kolom == (columns - 1) and not best_piece.get_edges()[
                                                                         2].get_type() == 'straight') or
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
                                     (rij != 0 and pieces_solved[(number + 1) - columns].get_edges()[1].get_type() ==
                                      best_piece.get_edges()[3].get_type()))):
                # Terugzetten naar de originele toestand zoals ze in pieces_copy staan
                best_piece.rotate(360 - rotate_angle)
                lijst_van_index_foute_pieces_en_randen.append((pieces_copy.index(best_piece), best_piece_edge_number))
                best_piece = None
                best_piece_edge_number = None
                best_match_value = 100000
            else:
                logicIsOk = True

        pieces_solved.append(best_piece)
        pieces_copy.remove(best_piece)

    # cv2.imshow('laatste piece', pieces_solved[len(pieces_solved)-1].get_piece())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pieces_solved = np.array(pieces_solved).reshape(rows, columns)

    solved_image = np.zeros(
        [pieces_solved[0][0].get_height() * rows + 10, pieces_solved[0][0].get_width() * columns + 10, 3],
        dtype=np.uint8)

    width_uitsteek = 0
    height_uitsteek = 0
    for r, row_pieces in enumerate(pieces_solved):
        for c, piece in enumerate(row_pieces):
            if ((piece.get_edges()[0].get_type() == "innie" and piece.get_edges()[2].get_type() == "outie") or
                    (piece.get_edges()[0].get_type() == "outie" and piece.get_edges()[2].get_type() == "innie") or
                    (piece.get_edges()[0].get_type() == "straight" and piece.get_edges()[2].get_type() == "outie") or
                    (piece.get_edges()[0].get_type() == "outie" and piece.get_edges()[2].get_type() == "straight")):
                width_uitsteek = piece.get_piece_width() - piece.get_width()
            if ((piece.get_edges()[1].get_type() == "innie" and piece.get_edges()[3].get_type() == "outie") or
                    (piece.get_edges()[1].get_type() == "outie" and piece.get_edges()[3].get_type() == "innie") or
                    (piece.get_edges()[1].get_type() == "straight" and piece.get_edges()[3].get_type() == "outie") or
                    (piece.get_edges()[1].get_type() == "outie" and piece.get_edges()[3].get_type() == "straight")):
                height_uitsteek = piece.get_piece_height() - piece.get_height()

    min_y = 0
    max_x = 0
    totaal_y = 0
    for r, row_pieces in enumerate(pieces_solved):
        for c, piece in enumerate(row_pieces):
            width_uitsteek = get_offset_x(piece, width_uitsteek)
            if c > 0:
                min_x = max_x - width_uitsteek - 1
            else:
                min_x = 0

            height_uitsteek = get_offset_y(piece, height_uitsteek)
            if r > 0:
                if pieces_solved[r - 1][c].get_edges()[1].get_type() == "outie":
                    min_y = totaal_y
                else:
                    min_y = totaal_y - height_uitsteek-1
            max_x = min_x + piece.get_piece_width()
            max_y = min_y + piece.get_piece_height()

            temp_img = np.zeros_like(solved_image)
            temp_img[min_y:max_y, min_x:max_x, :] = piece.get_piece()
            solved_image = cv2.bitwise_or(solved_image, temp_img, mask=None)

        totaal_y += pieces_solved[r][0].get_height()

    return solved_image


def get_offset_x(piece, last_width_uitsteek):
    if piece.get_edges()[0].get_type() == "outie" and piece.get_edges()[2].get_type() == "outie":
        width_uitsteek = int(round((piece.get_piece_width() - piece.get_width()) / 2, 0))
        return width_uitsteek
    elif ((piece.get_edges()[0].get_type() == "innie" and piece.get_edges()[2].get_type() == "outie") or
          (piece.get_edges()[0].get_type() == "outie" and piece.get_edges()[2].get_type() == "innie") or
          (piece.get_edges()[0].get_type() == "straight" and piece.get_edges()[2].get_type() == "outie") or
          (piece.get_edges()[0].get_type() == "outie" and piece.get_edges()[2].get_type() == "straight")):
        width_uitsteek = (piece.get_piece_width() - piece.get_width())
        return width_uitsteek
    else:
        return last_width_uitsteek


def get_offset_y(piece, last_height_uitsteek):
    if piece.get_edges()[1].get_type() == "outie" and piece.get_edges()[3].get_type() == "outie":
        height_uitsteek = int(round((piece.get_piece_height() - piece.get_height()) // 2, 0))
        return height_uitsteek
    elif ((piece.get_edges()[1].get_type() == "innie" and piece.get_edges()[3].get_type() == "outie") or
          (piece.get_edges()[1].get_type() == "outie" and piece.get_edges()[3].get_type() == "innie") or
          (piece.get_edges()[1].get_type() == "straight" and piece.get_edges()[3].get_type() == "outie") or
          (piece.get_edges()[1].get_type() == "outie" and piece.get_edges()[3].get_type() == "straight")):
        height_uitsteek = (piece.get_piece_height() - piece.get_height())
        return height_uitsteek
    else:
        return last_height_uitsteek
