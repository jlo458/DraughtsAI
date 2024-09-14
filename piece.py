import pygame

from .consts import BLACK, CROWN, WHITE


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
        pygame.draw.circle(window, self.colour, (self.x, self.y), 35)
        if self.king:
            window.blit(CROWN, (self.x - CROWN.get_width() / 2, self.y - CROWN.get_height() / 2))

    def move(self, row, col):
        self.row = row
        self.col = col
        self.position()
