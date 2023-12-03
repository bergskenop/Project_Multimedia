import numpy as np
class Edge:
    def __init__(self, hoeken, points=None):
        self.edge_points = points
        self.type = None
        self.hoeken = hoeken

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
        self.edge_points = points
    def print_edge(self):
        print(f'Hoeken : {self.hoeken} van het type {self.type} met {len(self.edge_points)} punten')
