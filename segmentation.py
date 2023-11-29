import cv2

def process_puzzle(puzzle):
    puzzle_contours = get_puzzle_contours(puzzle.image, puzzle.rows, puzzle.columns)
    if len(puzzle_contours) == puzzle.size:
        print(f"Successfully found {puzzle.size} puzzle pieces")
    else:
        raise Exception("Given number of pieces has not been found")
    if puzzle.type == 2:  # Check if puzzle is of scrambled type
        rotate_piece()
    return puzzle_contours


def get_puzzle_contours(img, r, c):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.blur(imgray,(5,5))
    ret, thresh = cv2.threshold(imgray, 0, 254, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours[:r*c]


def rotate_piece(): #rotate piece around centre until straight edge is alligned

    return 0

def retrieve_corners(img, contours):
    print(len(contours))
    for contour in contours:
        x_values = contour[:, 0, 0]
        y_values = contour[:, 0, 1]

        #
        # # Finding top left, top right, bottom left, and bottom right points
        # top_left = [min(x_values), min(y_values)]
        # top_right = [max(x_values), min(y_values)]
        # bottom_left = [min(x_values), max(y_values)]
        # bottom_right = [max(x_values), max(y_values)]
        #
        # cv2.circle(img, tuple(top_left), 5, (255, 0, 0), -1)  # Draw top left point
        # cv2.circle(img, tuple(top_right), 5, (255, 0, 0), -1)  # Draw top right point
        # cv2.circle(img, tuple(bottom_left), 5, (255, 0, 0), -1)  # Draw bottom left point
        # cv2.circle(img, tuple(bottom_right), 5, (255, 0, 0), -1)  # Draw bottom right point
        #
        cv2.imshow('points', img)
        cv2.waitKey(0)
