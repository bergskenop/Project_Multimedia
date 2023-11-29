import numpy as np
from matplotlib import pyplot as plt
import cv2
import re
from tkinter import filedialog
from puzzel_parameters import *
from segmentation import *


def main():
    path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                      filetypes=(("Images", "*.png*"), ("all files", "*.*")))
    type_puzzel, aantal_rijen, aantal_kolommen = bepaal_puzzel_parameters(path)
    print(type_puzzel, aantal_rijen, aantal_kolommen)
    process_puzzle(path, type_puzzel, aantal_rijen*aantal_kolommen)


if __name__ == '__main__':
    main()
