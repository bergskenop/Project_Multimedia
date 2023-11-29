class PuzzlePiece:
    def __init__(self, points, corners):
        self.points = points
        self.corners = corners
        self.edges = None

    def set_points(self, points):
        self.points = points

    def set_corners(self, corners):
        self.corners = corners

    def set_edges(self, edges):
        self.edges = edges
