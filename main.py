from tkinter import filedialog
from puzzel_parameters import *
from segmentation import *
import os
from Puzzle import Puzzle
import cv2

directory = "data/"

def main():
    path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                      filetypes=(("Images", "*.png*"), ("all files", "*.*")))
    type_puzzel, aantal_rijen, aantal_kolommen = bepaal_puzzel_parameters(path)
    p = Puzzle(path, type_puzzel, aantal_rijen, aantal_kolommen)
    p.set_contours()
    p.show()
    print(p.contourCorners)


def process_all():
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(subdir, file)
            type_puzzel, aantal_rijen, aantal_kolommen = bepaal_puzzel_parameters(path)
            process_puzzle(path, type_puzzel, aantal_rijen*aantal_kolommen)

if __name__ == '__main__':
    main()
