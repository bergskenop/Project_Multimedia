class Edge:
    def __init__(self, hoeken):
        self.type = None
        self.hoeken = hoeken

    def set_type(self, type):
        self.type = type

    def set_hoeken(self, hoeken):
        self.hoeken = hoeken

    def print_edge(self):
        for i, hoek in enumerate(self.hoeken):
            print(f'Hoek {i}: {hoek} van het type {self.type}')
