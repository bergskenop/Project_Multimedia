import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class Edge:
    def __init__(self, hoeken, points=None):
        self.edge_points = points
        self.type = None
        self.hoeken = hoeken

    # Set_type gaat er steeds van uit dat het eerste hoekpunt zich linksboven bevind
    # edge_number variabele bepaald in welke richting de hoekpunten liggen
    # (kan ook zonder door hoeken variabele te analyseren)
    def set_type(self, edge_number, width, height):
        # 1 = straight; 2 = innie; 3 = outie
        if edge_number == 0:
            if len(self.edge_points) <= height+1:
                self.type = 'straight'
            elif np.any([x < self.hoeken[0][0] for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        elif edge_number == 1:
            if len(self.edge_points) <= width+1:
                self.type = 'straight'
            elif np.any([y > self.hoeken[1][1] for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        elif edge_number == 2:
            if len(self.edge_points) <= height+1:
                self.type = 'straight'
            elif np.any([x < self.hoeken[0][0] for x, y in self.edge_points]):
                self.type = 'innie'
            else:
                self.type = 'outie'
        else:
            if len(self.edge_points) <= width+1:
                self.type = 'straight'
            elif np.any([y < self.hoeken[1][1] for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        return 0

    def set_hoeken(self, hoeken):
        self.hoeken = hoeken

    def set_points(self, points):
        # Use Linear and cubic spline to interpolate edges
        self.edge_points = points

    def get_points(self):
        return self.points

    def print_edge(self):
        print(f'Hoeken : {self.hoeken} van het type {self.type} met {len(self.edge_points)} punten')

    # Current edge point are so few as possible to draw the correct edge, but if we want to calculate the correct
    # histogram we want alle points along the edge, for that we draw the edge with the few edge points and we apply
    # masking to detect all the points on the edge. Afterwards we calculate the color histogram of the edge so we can
    # use it to match with other edges later.
    def detect_all_edge_points(self, image):
        return 0
        # contour = [np.array(self.edge_points).reshape((len(self.edge_points), 1, 2))]
        # cv2.drawContours(image, contour, -1, (0, 255, 0), 1)
        # cv2.imshow("test", image)
        # cv2.waitKey(0)
