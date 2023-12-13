from tkinter import filedialog
import os
from Puzzle import Puzzle


def main():
    path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                      filetypes=(("Images", "*.png*"), ("all files", "*.*")))
    # path = 'data/Jigsaw_shuffled/jigsaw_shuffled_2x2_00.png'
    p = Puzzle(path)
    p.initialise_puzzle()
    # p.show()
    # p.draw_contours()
    # p.draw_corners()
    p.match()
    # p.type_based_matching()
    # p.show(p.solved_image)
    # process_all("data/Jigsaw_shuffled")


def process_all(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            print(file)
            path = os.path.join(subdir, file)
            puzzle = Puzzle(path)
            puzzle.initialise_puzzle()
            # puzzle.show()
            # puzzle.draw_contours()
            # puzzle.draw_corners()


if __name__ == '__main__':
    main()
