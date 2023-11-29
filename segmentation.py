import cv2
import numpy as np
from tkinter import filedialog


def process_puzzle(path, puzzle_orientation, n):
    puzzle = cv2.imread(path)
    puzzle_contours = get_puzzle_contours(puzzle, n)
    print(len(puzzle_contours))
    if len(puzzle_contours) == n:
        print(f"Successfully found {n} puzzle pieces")
    if puzzle_orientation == 3:
        rotate_piece()
    return 0


def get_puzzle_contours(img, number):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 250, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(0, number):
        cv2.drawContours(img, contours, i, (0, 255, 0), 3)
    cv2.imshow('contours', img)
    cv2.waitKey(0)
    return contours


def rotate_piece():
    return 0
