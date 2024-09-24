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
            moves.append(self.checkRight(row, col, colour, -1))
            moves.append(self.checkLeft(row, col, colour, -1))

        if colour == WHITE or piece.king: 
            moves.append(self.checkRight(row, col, colour, 1))
            moves.append(self.checkLeft(row, col, colour, 1))

        return moves


    def check(self, row, col): 
        moves = None
        piece = self.board.select_piece(row, col)
        if type(piece) is not int and piece.colour == self.go: 
            moves = self.possibleMoves(piece) 

        return moves


    def checkRight(self, row, col, colour, direction): 
        if col == 7 or (row+direction) > 7 or (row+direction) < 0: 
            return None   

        rightSpace = self.board.select_piece(row+direction, col+1)
        if rightSpace == 0:
            return row+direction, col+1
        
        elif rightSpace.colour == colour:
            return None 
        
        else:
            return self._takeRight(row, col, direction)

    def _takeRight(self, row, col, direction):
        if col >= 6 or row+(2*direction) > 7 or row+(2*direction) < 0: 
            return None
            
        rightSpace = self.board.select_piece(row+(2*direction), col+2)
        if rightSpace == 0:
            return row+(2*direction), col+2 
            
        return None 

    def checkLeft(self, row, col, colour, direction):  
        if col == 0 or (row+direction)>7 or (row+direction)<0: 
            return None 
    
        leftSpace = self.board.select_piece(row+direction, col-1)
        if leftSpace == 0:
            return row+direction, col-1
        
        elif leftSpace.colour == colour:
            return None 
        
        else:
            return self._takeLeft(row, col, direction)

    def _takeLeft(self, row, col, direction): 
        if col <= 1 or row+(2*direction) > 7 or row+(2*direction) < 0: 
            return None
        leftSpace = self.board.select_piece(row+(2*direction), col-2)
        if leftSpace == 0: 
            return row+(2*direction), col-2 
            
        return None

    def drawMoves(self, moves): 
        for move in moves:
            if move is not None: 
                row = move[1]
                col = move[0]
                if row < 8 and row > -1 and col > -1 and row < 8:
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
                self.validMoves = self.possibleMoves(piece)
                return True 
        
        return False
    
    def _move(self, row, col): 
        reMove = False
        if self.selected and (row, col) in self.validMoves:
            oldRow = self.selected.row 
            oldCol = self.selected.col
            self.board.move(self.selected, row, col)
            if self.selected.row == oldRow-2 or self.selected.row == oldRow+2:
                #print((oldRow+self.selected.row)//2, )
                self.board.removePiece((oldRow+self.selected.row)//2, (oldCol+self.selected.col)//2)
                moves = self.checkDouble(self.selected.colour)
                self.validMoves = []
                for move in moves: 
                    if move is not None: 
                        reMove = True
                        self.validMoves.append(move)

            if not reMove:
                self.changeTurn() # Change later for double takes

            if self.selected.row == 7 or self.selected.row == 0:
                self.selected.makeKing()

        else: 
            return False 
        
        return True

    def checkDoubleDirection(self, direction, row, col, colour, moves, piece): 
        try: 
            rightSpace = self.board.select_piece(row+direction, col+1)
            if rightSpace.colour != colour:
                moves.append(self._takeRight(piece.row, piece.col, direction))
            
        except: 
            pass

        try: 
            leftSpace = self.board.select_piece(row+direction, col-1)
            if leftSpace.colour != colour:
                moves.append(self._takeLeft(piece.row, piece.col, direction))
            
        except: 
            pass

        return moves

    def checkDouble(self, colour):
        moves = []
        piece = self.selected
        row = piece.row
        col = piece.col
        if colour == BLACK or piece.king:
            direction = -1
            moves = self.checkDoubleDirection(direction, row, col, colour, moves, piece)

        if colour == WHITE or piece.king: 
            direction = 1
            moves = self.checkDoubleDirection(direction, row, col, colour, moves, piece)

        return moves

    def changeTurn(self): 
        self.validMoves = []
        if self.go == BLACK: 
            self.go = WHITE 

        else: 
            self.go = BLACK
