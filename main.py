from tkinter import filedialog
import os
from Puzzle import Puzzle


def main():
    path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                      filetypes=(("Images", "*.png*"), ("all files", "*.*")))
    p = Puzzle(path)
    p.initialise_puzzle()
    # p.show()
    # p.draw_contours()
    p.draw_corners()
    # process_all("data/")


def process_all(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(subdir, file)
            puzzle = Puzzle(path)
            puzzle.initialise_puzzle()
            puzzle.show()
            puzzle.draw_contours()
            puzzle.draw_corners()


if __name__ == '__main__':
    main()
