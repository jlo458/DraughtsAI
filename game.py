# A bit of a mess right now, should be a bot cleaner later on this week

import pygame

from .board import Board
from .consts import BLACK, WHITE
from .piece import Piece


class Game: 
    def __init__(self, window):
        self._startGame()
        self.window = window

    def update(self): 
        self.board.drawAll(self.window)
        pygame.display.update()

    def _startGame(self): 
        self.selected = None
        self.board = Board()
        self.go = BLACK
        self.validMoves = {}

    def reset(self): 
        self._startGame()

    def possibleMoves(self, row, col):
        pass

    def check(self, row, col): 
        piece = self.board.select_piece(row, col)
        if type(piece) is not int and piece.colour == self.go: 
            pass 


        
    def select(self, row, col): 
        if self.selected: 
            move = self._move(row, col)
            if not move: 
                self.selected = None


    def _move(self): 
        pass
