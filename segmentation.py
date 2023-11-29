import cv2
import numpy as np

def process_puzzle(puzzle):
    puzzle_contours = get_puzzle_contours(puzzle.image, puzzle.rows, puzzle.columns)
    hoekpunten = retrieve_corners(puzzle.image, puzzle_contours, puzzle.columns, puzzle.rows)
    return puzzle_contours


def get_puzzle_contours(img, r, c):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 254, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[:r*c]


def rotate_piece(): #rotate piece around centre until straight edge is alligned

    return 0


def retrieve_corners(img, contours):
    hoekpunten = []
    for contour in contours:
        cont = contour.reshape((contour.shape[0], contour.shape[2]))
        cont = np.vstack([cont, [10, 10]])
        x_all = cont[:, 0]
        y_all = cont[:, 1]

        diff = np.diff(cont, axis=0)
        distances = np.linalg.norm(diff, axis=1)
        # Later robuust maken
        # min_dist = img.shape[0] / aantal_rijen / 2 * 0.6
        # if img.shape[1] / aantal_kolommen / 2 * 0.7:
        #     min_dist = img.shape[1] / aantal_kolommen / 2 * 0.6
        indices_greater_than_min_dist = np.where(distances > 10)
        x_nieuw = []
        y_nieuw = []
        for i in indices_greater_than_min_dist:
            x_nieuw.append(x_all[i])
            y_nieuw.append(y_all[i])
        x_nieuw = np.squeeze(np.array(x_nieuw))
        y_nieuw = np.squeeze(np.array(y_nieuw))
        unique_elements, counts = np.unique(x_nieuw, return_counts=True)
        elements_to_remove = unique_elements[counts == 1]
        for i in elements_to_remove:
            index = np.where(x_nieuw == i)
            x_nieuw = np.delete(x_nieuw, index)
            y_nieuw = np.delete(y_nieuw, index)

        unique_elements, counts = np.unique(y_nieuw, return_counts=True)
        elements_to_remove = unique_elements[counts == 1]
        for i in elements_to_remove:
            index = np.where(y_nieuw == i)
            x_nieuw = np.delete(x_nieuw, index)
            y_nieuw = np.delete(y_nieuw, index)

        for i in range(len(x_nieuw)):
            hoekpunten.append((x_nieuw[i], y_nieuw[i]))

    return hoekpunten
