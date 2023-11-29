import numpy as np
from matplotlib import pyplot as plt
import cv2
import re
from tkinter import filedialog


def bepaal_puzzel_parameters(image_path):
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.show()

    # 1 = shuffled, 2 = scrambled and 3 = rotated
    type_puzzle = 1
    if re.search(".+_scrambled_.+", image_path):
        type_puzzle = 2
    elif re.search(".+_rotated_.+", image_path):
        type_puzzle = 3

    # Bepaal aantal rijen en kolommen
    scale = re.compile("[0-9][x][0-9]").findall(image_path)
    rijen = int(str(re.compile("^[0-9]").findall(scale[0])[0]))
    kolommen = int(str(re.compile("[0-9]$").findall(scale[0])[0]))

    return type_puzzle, rijen, kolommen



