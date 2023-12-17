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
    p.draw_contours()
    p.draw_corners()
    p.match()
    # p.show(p.solved_image)
    # process_all("data/Jigsaw_shuffled")
    # process_all("data/Jigsaw_rotated")
    # process_all("data/Jigsaw_scrambled")


def process_all(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if (file != "jigsaw_rotated_5x5_01.png" and
                    file != "jigsaw_rotated_5x5_03.png" and file != "jigsaw_rotated_5x5_06.png" and
                    file != "jigsaw_shuffled_5x5_01.png" and
                    file != "jigsaw_shuffled_5x5_03.png" and file != "jigsaw_shuffled_5x5_06.png" and
                    file != "jigsaw_scrambled_5x5_01.png" and
                    file != "jigsaw_scrambled_5x5_03.png" and file != "jigsaw_scrambled_5x5_06.png"):
                print(file)
                path = os.path.join(subdir, file)
                puzzle = Puzzle(path)
                puzzle.initialise_puzzle()
                # puzzle.show()
                # puzzle.draw_contours()
                # puzzle.draw_corners()
                puzzle.match()


if __name__ == '__main__':
    main()
