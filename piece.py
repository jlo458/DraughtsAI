import pygame
from .consts import BLACK, WHITE


class Piece:
    def __init__(self, row, col, colour):
        self.row = row
        self.col = col
        self.colour = colour
        self.king = False

        if self.colour == WHITE:
            self.direction = 1

        else:
            self.direction = -1

        self.x = 0
        self.y = 0
        self.position()

    def position(self):
        self.x = 100 * (self.col) + 50
        self.y = 100 * (self.row) + 50

    def makeKing(self):
        self.King = True

    def drawPiece(self, window):
        pygame.draw.circle(window, self.colour, (self.x, self.y), 40)
