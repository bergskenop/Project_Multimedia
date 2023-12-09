from Edge import *
import cv2

class PuzzlePiece:
    def __init__(self, points, corners):
        self.piece = None
        self.points = points
        self.corners = corners
        self.edges = []

    def set_piece(self, image):
        self.piece = image

    def get_piece(self):
        return self.piece

    def set_edges(self, width, height, image):
        for i in range(3):
            eerste_index = self.points.index(self.corners[i])
            laatste_index = self.points.index(self.corners[i+1])
            self.edges.append(Edge((self.corners[i], self.corners[i+1]), self.points[eerste_index:laatste_index+1]))
        punten_laatste_rand = (self.points[self.points.index(self.corners[3]):] +
                               self.points[:self.points.index(self.corners[0])+1])
        self.edges.append(Edge((self.corners[3], self.corners[0]), punten_laatste_rand))

        for i, edge in enumerate(self.edges):
            edge.set_type(i, width, height)
            edge.calculate_histogram(image)
            # edge.print_edge()

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
