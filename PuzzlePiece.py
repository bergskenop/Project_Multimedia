class PuzzlePiece:
    def __init__(self, points, corners):
        self.points = points
        self.corners = corners
        self.edges = None

    def set_edges(self):
        self.edges = 0
