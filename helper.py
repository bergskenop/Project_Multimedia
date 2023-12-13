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

def match(pieces, piece_dim, puzzle_dim):
    height_puzzle_piece, width_puzzle_piece = piece_dim
    rows, columns, _ = puzzle_dim
    solved_image = np.zeros([height_puzzle_piece * rows, width_puzzle_piece * columns, 3], dtype=np.uint8)
    # Begin puzzelstuk zoeken door het eerste puzzelstuk met 2 rechte lijnen te vinden en dit te draaien tot het
    # hoekpunt linksboven is zodat w esteeds van daaruit vertrekken bij het matchen van puzzelstukken
    i = 0
    corner_found = False
    while not corner_found:
        aantal_rechte_lijnen = 0
        for edge in pieces[i].get_edges():
            if edge.get_type() == 'straight':
                aantal_rechte_lijnen += 1
        if aantal_rechte_lijnen == 2:
            while not pieces[i].get_edges()[0].get_type() == 'straight' and not pieces[i].get_edges()[3].get_type() == 'straight':
                pieces[i]



