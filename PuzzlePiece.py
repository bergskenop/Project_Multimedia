from Edge import *
import numpy as np
import cv2


class PuzzlePiece:
    def __init__(self, points, corners):
        self.piece = None
        self.points = points
        self.corners = corners
        self.edges = []

    def set_piece(self, image):
        self.piece = image

    def set_edges(self):
        first_corner = self.corners[0]
        for i in range(1, len(self.corners)):
            self.edges.append(Edge((self.corners[i - 1], self.corners[i])))
        self.edges.append(Edge((self.corners[len(self.corners) - 1], first_corner)))

        for i in range(0, len(self.edges) - 1):
            self.edges[i].set_points(self.points[self.points.index(self.edges[i].hoeken[0]):
                                                 self.points.index(self.edges[i].hoeken[1])])
        self.edges[3].set_points(self.points[self.points.index(self.edges[3].hoeken[0]):])
        for i, edge in enumerate(self.edges):
            edge.set_type(i)

    def get_edges(self):
        return self.edges

    def get_points(self):
        return self.points

    def show_puzzlepiece(self):
        if self.piece is not None:
            cv2.imshow(f'Puzzlepiece', self.piece)
            cv2.waitKey(0)
            cv2.destroyWindow('Puzzlepiece')

    def print_puzzlepiece(self):
        print(f'Puzzelpiece heeft {len(self.points)} punten')
        print(f'Puzzelpiece heeft {len(self.corners)} hoeken')
        print(f'Puzzelpiece heeft {len(self.edges)} randen')
        for i, edge in enumerate(self.edges):
            print(f'Puzzelpiece heeft als randen:')
            edge.print_edge()
