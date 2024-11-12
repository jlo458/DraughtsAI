#from curses import window

import pygame

from .board import Board
from .consts import BLACK, GREY, WHITE
from .piece import Piece


class Game: 
    def __init__(self, window):
        self._startGame()
        self.window = window

    def update(self):
        try: 
            self.board.drawAll(self.window)
            try:
                self.drawMoves(self.validMoves)

            except: 
                pass
            pygame.display.update()

        except: 
            print(self.board)

    def _startGame(self): 
        self.selected = None
        self.board = Board()
        self.go = BLACK
        self.validMoves = []

    def reset(self): 
        self._startGame()


    def drawMoves(self, moves): 
        for move in moves:
            if move is not None: 
                row = move[1]
                col = move[0]
                if row < 8 and row > -1 and col > -1 and col < 8:
                    pygame.draw.circle(self.window, GREY, (row*100 + 50, col*100 + 50), 15)

    def select(self, row, col): 
        if self.selected: 
            move = self._move(row, col)
            if not move: 
                self.selected = None
                self.select(row, col)

        else:
            piece = self.board.select_piece(row, col)
            if piece != 0 and piece.colour == self.go: 
                self.selected = piece
                self.validMoves = self.board.possibleMoves(piece)
                return True 
        
        return False
    
    def _move(self, row, col): 
        reMove = False
        if self.selected and (row, col) in self.validMoves:
            oldRow = self.selected.row 
            oldCol = self.selected.col
            self.board.move(self.selected, row, col)
            if self.selected.row == oldRow-2 or self.selected.row == oldRow+2:
                self.board.removePiece((oldRow+self.selected.row)//2, (oldCol+self.selected.col)//2)
                moves = self.board.checkDouble(self.selected, self.selected.colour)
                self.validMoves = []
                for move in moves: 
                    if move is not None: 
                        reMove = True
                        self.validMoves.append(move)

            if not reMove:
                self.changeTurn() 

            '''if self.selected.row == 7 or self.selected.row == 0:
                self.selected.makeKing()'''

        else: 
            return False 
        
        return True

    def changeTurn(self): 
        self.validMoves = []
        if self.go == BLACK: 
            self.go = WHITE 

        else: 
            self.go = BLACK

    def getBoard(self):
        return self.board
    
    def minimaxMove(self, board): 
        self.board = board 
        self.changeTurn()
