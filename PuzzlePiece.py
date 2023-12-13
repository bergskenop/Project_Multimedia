from Edge import *
import cv2

class PuzzlePiece:
    def __init__(self, points):
        self.piece = None
        self.points = points
        self.corners = []
        self.edges = []
        self.width = None
        self.height = None

    def set_piece(self, image):
        self.piece = image

    def get_piece(self):
        return self.piece

    def set_width_and_height(self, width, height):
        self.width = width
        self.height = height

    def set_edges_and_corners(self, image, corners):
        # Hier corners instellen omdat de punten die Harris corner detection vindt niet altijd in de contour liggen,
        # Daarom dichtsbijzijnde punten in de contour vinden en deze gebruiken als hoekpunt
        for corner in corners:
            target_point = np.array(corner)
            points_array = np.array(self.points)
            distances = np.linalg.norm(points_array - target_point, axis=1)
            closest_index = np.argmin(distances)
            closest_point = tuple(points_array[closest_index])
            self.corners.append(closest_point)

        for i in range(3):
            eerste_index = self.points.index(self.corners[i])
            laatste_index = self.points.index(self.corners[i+1])
            self.edges.append(Edge((self.corners[i], self.corners[i+1]), self.points[eerste_index:laatste_index+1]))
        punten_laatste_rand = (self.points[self.points.index(self.corners[3]):] +
                               self.points[:self.points.index(self.corners[0])+1])
        self.edges.append(Edge((self.corners[3], self.corners[0]), punten_laatste_rand))

        width = abs(self.corners[1][0] - self.corners[2][0])
        height = abs(self.corners[0][1] - self.corners[1][1])
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
