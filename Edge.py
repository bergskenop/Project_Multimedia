import numpy as np
import cv2
import matplotlib.pyplot as plt


class Edge:
    def __init__(self, hoeken, points, lengte):
        self.edge_points = points  # Alle punten van de rand
        self.type = None  # Straight, innie of outie
        self.hoeken = hoeken  # 2 hoekpunten van de rand (2 uitersten)
        self.histogram = None  # Bevat de punten van het histogram van de zwart-wit waarden van de randen
        self.lengte = lengte


    # Set_type gaat er steeds van uit dat het eerste hoekpunt zich linksboven bevind
    # edge_number variabele bepaald in welke richting de hoekpunten liggen
    # (kan ook zonder door hoeken variabele te analyseren)
    def set_type(self, edge_number, size, type_puzzel):
        # Grotere speling nodig bij de hoeken van scrambled, hier gemakkelijk aan te passen,
        # bij shuffled en rotated is 3 al genoeg
        if size == 25:
            speling = 5
        else:
            speling = 8
        if edge_number == 0:
            if np.all([self.hoeken[0][0]-speling < x < self.hoeken[0][0]+speling for x, y in self.edge_points]):
                self.type = 'straight'
            elif np.any([x < self.hoeken[0][0]-speling for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        elif edge_number == 1:
            if np.all([self.hoeken[0][1]-speling < y < self.hoeken[0][1]+speling for x, y in self.edge_points]):
                self.type = 'straight'
            elif np.any([y > self.hoeken[1][1]+speling for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        elif edge_number == 2:
            if np.all([self.hoeken[0][0]-speling < x < self.hoeken[0][0]+speling for x, y in self.edge_points]):
                self.type = 'straight'
            elif np.any([x < self.hoeken[0][0]-speling for x, y in self.edge_points]):
                self.type = 'innie'
            else:
                self.type = 'outie'
        else:
            if np.all([self.hoeken[0][1]-speling < y < self.hoeken[0][1]+speling for x, y in self.edge_points]):
                self.type = 'straight'
            elif np.any([y < self.hoeken[1][1]-speling for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        return 0

    def set_lengte(self, lengte):
        self.lengte = lengte

    def get_lengte(self):
        return self.lengte

    def set_hoeken(self, hoeken):
        self.hoeken = hoeken

    def set_points(self, points):
        self.edge_points = points

    def get_edge_points(self):
        return self.edge_points

    def get_type(self):
        return self.type

    def get_histogram(self):
        return self.histogram

    def print_edge(self):
        print(f'Hoeken : {self.hoeken} van het type {self.type} met {len(self.edge_points)} punten')


    # Calculate the histogram of the edge points in an image
    # Omdat cv2.circle de punten tekent op de rand van de figuur zullen de helft van de randpunten in het pikzwarte
    # gebied liggen, vandaar doen we nog eens een bitwise and met de puzzel stukken om alleen de echte randpixels
    # over te houden. We kunnen de radius van de cirkelpunten vergroten of verkleinen als we meer of minder
    # randpunten zouden willen bekijken
    def calculate_histogram(self, image):
        mask = np.zeros_like(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image_threshold = cv2.threshold(gray_image, 0, 254, cv2.THRESH_BINARY)
        for point in self.edge_points:
            cv2.circle(mask, point, 4, (255, 255, 255), -1)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_perfect = cv2.bitwise_and(image_threshold, image_threshold, mask=gray_mask)
        hist = cv2.calcHist([gray_image], [0], mask_perfect, [256], [0, 256])
        self.histogram = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # plt.plot(self.histogram)
        # plt.show()

