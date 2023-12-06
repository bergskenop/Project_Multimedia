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
    def set_type(self, edge_number):
        # 1 = straight; 2 = innie; 3 = outie
        if len(self.edge_points) < 10:
            self.type = 'straight'
        elif edge_number == 0:
            if np.any([x < self.hoeken[0] for x, y in self.edge_points]):
                self.type = 'outie'
            elif np.any([x > self.hoeken[0] for x, y in self.edge_points]):
                self.type = 'innie'
        elif edge_number == 1:
            if np.any([y > self.hoeken[1] for x, y in self.edge_points]):
                self.type = 'outie'
            elif np.any([y < self.hoeken[1] for x, y in self.edge_points]):
                self.type = 'innie'
        elif edge_number == 2:
            if np.any([x < self.hoeken[0] for x, y in self.edge_points]):
                self.type = 'innie'
            elif np.any([x > self.hoeken[0] for x, y in self.edge_points]):
                self.type = 'outie'
        else:
            if np.any([y < self.hoeken[1] for x, y in self.edge_points]):
                self.type = 'outie'
            elif np.any([y > self.hoeken[1] for x, y in self.edge_points]):
                self.type = 'innie'
        return 0


    def set_hoeken(self, hoeken):
        self.hoeken = hoeken

    def set_points(self, points):
        # Use Linear and cubic spline to interpolate edges
        self.edge_points = points

    def get_points(self):
        return self.points
    def set_descriptors(self):
        return 0

    def print_edge(self):
        print(f'Hoeken : {self.hoeken} van het type {self.type} met {len(self.edge_points)} punten')
