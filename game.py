# Now be able to move pieces

import pygame
from .board import Board
from .consts import BLACK, GREY, WHITE
from .piece import Piece


class Game: 
    def __init__(self, window):
        self._startGame()
        self.window = window

    def update(self): 
        self.board.drawAll(self.window)
        try:
            self.drawMoves(self.validMoves)

        except: 
            pass
        pygame.display.update()

    def _startGame(self): 
        self.selected = None
        self.board = Board()
        self.go = BLACK
        self.validMoves = []

    def reset(self): 
        self._startGame()

    def possibleMoves(self, piece):
        moves = []
        row = piece.row 
        col = piece.col 
        colour = piece.colour

        if colour == BLACK or piece.king: 
            moves.append(self.checkRight(row, col, colour))
            moves.append(self.checkLeft(row, col, colour))


        if colour == WHITE or piece.king: 
            moves.append(self.checkRight(row, col, colour))
            moves.append(self.checkLeft(row, col, colour))

        return moves


    def check(self, row, col): 
        moves = None
        piece = self.board.select_piece(row, col)
        if type(piece) is not int and piece.colour == self.go: 
            moves = self.possibleMoves(piece) 

        return moves


    def checkRight(self, row, col, colour): 
        if col == 7: 
            return None 
        
        direction = 1
        if colour == BLACK: # Check that this works
            direction = -1  

        rightSpace = self.board.select_piece(row+direction, col+1)
        if rightSpace == 0:
            return row+direction, col+1
        
        elif rightSpace.colour == colour:
            return None 
        
        else:
            if col == 1: 
                return None
            
            rightSpace = self.board.select_piece(row+(2*direction), col+2)
            if rightSpace == 0: 
                return row+(2*direction), col+2 
            
            return None 

    def checkLeft(self, row, col, colour): 
        if col == 0: 
            return None 
        
        #moves = {} 
        direction = 1
        if colour == BLACK: # Check that this works
            direction = -1  

        leftSpace = self.board.select_piece(row+direction, col-1)
        if leftSpace == 0:
            return row+direction, col-1
        
        elif leftSpace.colour == colour:
            return None 
        
        else:
            if col == 1: 
                return None
            
            leftSpace = self.board.select_piece(row+(2*direction), col-2)
            if leftSpace == 0: 
                return row+(2*direction), col-2 
            
            return None
            

    
  
    def select(self, row, col): 
        if self.selected: 
            move = self._move(row, col)
            if not move: 
                self.selected = None

    def drawMoves(self, moves): 
        for move in moves: 
            row = move[1]
            col = move[0]
            pygame.draw.circle(self.window, GREY, (row*100 + 50, col*100 + 50), 15)

    def _move(self): 
        pass

    def changeTurn(self): 
        if self.turn == BLACK: 
            self.turn == WHITE 

        else: 
            self.turn == BLACK
