import cv2
import numpy as np
from tkinter import filedialog

def segment_pieces(path):
    img = cv2.imread(path)
    cv2.imshow('image cut top', img[:img.shape[0] // 2])
    cv2.imshow('image cut bottom', img[img.shape[0] // 2:])
    cv2.waitKey(0)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 250, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
    cv2.drawContours(img, contours, 1, (255, 0, 0), 3)
    cv2.drawContours(img, contours, 2, (255, 255, 0), 3)
    cv2.drawContours(img, contours, 3, (0, 255, 255), 3)
    print(contours[0][0])
    cv2.imshow('contours', img)
    cv2.waitKey(0)
