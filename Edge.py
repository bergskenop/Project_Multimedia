import numpy as np
import cv2
import matplotlib.pyplot as plt


class Edge:
    def __init__(self, hoeken, points=None):
        self.edge_points = points  # Alle punten van de rand => eventueel verwijderen later
        self.type = None  # Straight, innie of outie
        self.hoeken = hoeken  # 2 hoekpunten van de rand (2 uitersten) => eventueel verwijderen later
        self.histogram = None  # Bevat de punten van het histogram van de zwart-wit waarden van de randen

    # Set_type gaat er steeds van uit dat het eerste hoekpunt zich linksboven bevind
    # edge_number variabele bepaald in welke richting de hoekpunten liggen
    # (kan ook zonder door hoeken variabele te analyseren)
    def set_type(self, edge_number, width, height):
        if edge_number == 0:
            if len(self.edge_points) <= height+(height//10):
                self.type = 'straight'
            elif np.any([x < self.hoeken[0][0] for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        elif edge_number == 1:
            if len(self.edge_points) <= width+(width//10):
                self.type = 'straight'
            elif np.any([y > self.hoeken[1][1] for x, y in self.edge_points]):
                self.type = 'outie'
            else:
                self.type = 'innie'
        elif edge_number == 2:
            if len(self.edge_points) <= height+(height//10):
                self.type = 'straight'
            elif np.any([x < self.hoeken[0][0] for x, y in self.edge_points]):
                self.type = 'innie'
            else:
                self.type = 'outie'
        else:
            if len(self.edge_points) <= width+(width//10):
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

    def get_type(self):
        return self.type

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
        # UNCOMMENT TO SHOW EDGES ONE BY ONE
        # cv2.imshow('mask', mask_perfect)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        hist = cv2.calcHist([gray_image], [0], mask_perfect, [256], [0, 256])
        hist_normalized = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        self.histogram = hist_normalized





