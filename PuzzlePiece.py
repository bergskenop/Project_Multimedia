from Edge import *


class PuzzlePiece:
    def __init__(self, points, corners):
        self.points = points
        self.corners = corners
        self.edges = []

    def set_edges(self):
        first_corner = self.corners[0]
        for i in range(1, len(self.corners)):
            self.edges.append(Edge((self.corners[i - 1], self.corners[i])))
        self.edges.append(Edge((self.corners[len(self.corners) - 1], first_corner)))
        for edge in self.edges:
            # Implement functionality to determine edge type: inward, outward or flat
            edge.set_type(None)

    def print_puzzlepiece(self):
        print(f'Puzzelpiece heeft {len(self.points)} punten')
        print(f'Puzzelpiece heeft {len(self.corners)} hoeken')
        print(f'Puzzelpiece heeft {len(self.edges)} randen')
        for i, edge in enumerate(self.edges):
            print(f'Puzzelpiece heeft als randen:')
            edge.print_edge()

