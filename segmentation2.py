import numpy as np
from matplotlib import pyplot as plt
import cv2
import re
from tkinter import filedialog
from puzzel_parameters import *

path = filedialog.askopenfilename(initialdir="*/", title="Select image",
                                      filetypes=(("Images", "*.png*"), ("all files", "*.*")))

type_puzzel, aantal_rijen, aantal_kolommen = bepaal_puzzel_parameters(path)

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# vind de randen
imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(imgray, 0, 250, 0)
plt.imshow(thresh, cmap='gray')
plt.show()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
cv2.drawContours(img, contours, 1, (255, 0, 0), 3)
cv2.drawContours(img, contours, 2, (255, 255, 0), 3)
cv2.drawContours(img, contours, 3, (0, 255, 255), 3)
plt.imshow(img)
plt.show()

for i in range(aantal_kolommen*aantal_rijen):
    cont = contours[i].reshape((contours[i].shape[0], contours[0].shape[2]))
    cont = np.vstack([cont, [10, 10]])
    print(cont)
    print(cont.shape)
    x_all = cont[:, 0]
    y_all = cont[:, 1]

    diff = np.diff(cont, axis=0)
    distances = np.linalg.norm(diff, axis=1)
    min_dist = img.shape[0] / aantal_rijen / 2 * 0.6
    if img.shape[1] / aantal_kolommen / 2 * 0.7:
        min_dist = img.shape[1] / aantal_kolommen / 2 * 0.6
    indices_greater_than_min_dist = np.where(distances > 10)
    x_nieuw = []
    y_nieuw = []
    for i in indices_greater_than_min_dist:
        x_nieuw.append(x_all[i])
        y_nieuw.append(y_all[i])
    x_nieuw = np.squeeze(np.array(x_nieuw))
    y_nieuw = np.squeeze(np.array(y_nieuw))
    unique_elements, counts = np.unique(x_nieuw, return_counts=True)
    elements_to_remove = unique_elements[counts == 1]
    for i in elements_to_remove:
        index = np.where(x_nieuw == i)
        x_nieuw = np.delete(x_nieuw, index)
        y_nieuw = np.delete(y_nieuw, index)

    unique_elements, counts = np.unique(y_nieuw, return_counts=True)
    elements_to_remove = unique_elements[counts == 1]
    for i in elements_to_remove:
        index = np.where(y_nieuw == i)
        x_nieuw = np.delete(x_nieuw, index)
        y_nieuw = np.delete(y_nieuw, index)

    plt.plot(x_nieuw, y_nieuw, marker='o', linestyle='')
    plt.imshow(imgray, cmap="gray")
plt.show()




