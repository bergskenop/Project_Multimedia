import numpy as np
from matplotlib import pyplot as plt
import cv2
import re
from tkinter import filedialog
from puzzel_parameters import *


def main():
    type_puzzel, aantal_rijen, aantal_kolommen = bepaal_puzzel_parameters()
    print(type_puzzel, aantal_rijen, aantal_kolommen)


if __name__ == '__main__':
    main()
