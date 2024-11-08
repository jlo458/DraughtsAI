import pygame

from .consts import BLACK, BLUE, WHITE
from .piece import Piece


class Board:
    def __init__(self) -> None:
        self.board = []
        self.chosenPiece = None
        self.blackLeft = self.whiteLeft = 12
        self.blackKings = self.whiteKings = 0
        self.buildBoard()

    def drawBoard(self, window):
        window.fill(WHITE)
        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 1:
                    pygame.draw.rect(window, BLUE, [100 * x, 100 * y, 100, 100])

        pygame.display.flip()

    def buildBoard(self):
        for row in range(8):
            self.board.append([])
            for col in range(8):
                if (row + col) % 2 == 1:
                    if row < 3:
                        self.board[row].append(Piece(row, col, WHITE))

                    elif row > 4:
                        self.board[row].append(Piece(row, col, BLACK))

                    else:
                        self.board[row].append(0)

                else:
                    self.board[row].append(0)

    def select_piece(self, row, col):
        return self.board[row][col]

    def move(self, piece, row, col):        
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)

        if row == 0 or row == 7:
            piece.makeKing()
            if piece.colour == WHITE:
                self.whiteKings += 1

            else:
                self.blackKings += 1

    def drawAll(self, window):
        self.drawBoard(window)
        for row in range(8):
            for col in range(8):
                object = self.board[row][col]
                if object != 0:
                    object.drawPiece(window)

    def removePiece(self, row, col):
        piece = self.board[row][col]
        if piece.colour == BLACK: 
            self.blackLeft -= 1
            self.checkWinner() 

        else: 
            self.whiteLeft -= 1 
            self.checkWinner()

        self.board[row][col] = 0

    def checkWinner(self): 
        if self.blackLeft == 0: 
            print("White Wins!")
            return True

        elif self.whiteLeft == 0: 
            print("Black Wins!")
            return True 
        
        return False

    def evaluateFunc(self): # Edit as you improve # Make more to check for best one
        w1 = 2
        w2 = 1
        score = w1 * (self.whiteLeft - self.blackLeft) + w2 * (self.whiteKings - self.blackKings)

        return score

    def getAllPieces(self, colour):
        validPieces = [] 
        for row in self.board:
            for piece in row: 
                if piece != 0 and piece.colour == colour: 
                    validPieces.append(piece)

        return validPieces
    
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

    def checkRight(self, row, col, colour, direction): 
        if col == 7 or (row+direction) > 7 or (row+direction) < 0: 
            return None   

        rightSpace = self.select_piece(row+direction, col+1)
        if rightSpace == 0:
            return row+direction, col+1
        
        elif rightSpace.colour == colour:
            return None 
        
        else:
            return self._takeRight(row, col, direction)

    def _takeRight(self, row, col, direction):
        if col >= 6 or row+(2*direction) > 7 or row+(2*direction) < 0: 
            return None
            
        rightSpace = self.select_piece(row+(2*direction), col+2)
        if rightSpace == 0:

            return row+(2*direction), col+2 
            
        return None 

    def checkLeft(self, row, col, colour, direction):  
        if col == 0 or (row+direction)>7 or (row+direction)<0: 
            return None 
        

        leftSpace = self.select_piece(row+direction, col-1)
        if leftSpace == 0:
            return row+direction, col-1
        
        elif leftSpace.colour == colour:
            return None 
        
        else:
            return self._takeLeft(row, col, direction)

    def _takeLeft(self, row, col, direction): 
        if col <= 1 or row+(2*direction) > 7 or row+(2*direction) < 0: 
            return None
        leftSpace = self.select_piece(row+(2*direction), col-2)
        if leftSpace == 0: 
            return row+(2*direction), col-2 
            
        return None

    def checkDoubleDirection(self, direction, row, col, colour, moves, piece): 
        try: 
            rightSpace = self.select_piece(row+direction, col+1)
            if rightSpace.colour != colour:
                moves.append(self._takeRight(piece.row, piece.col, direction))
            
        except: 
            pass

        try: 
            leftSpace = self.select_piece(row+direction, col-1)
            if leftSpace.colour != colour:
                moves.append(self._takeLeft(piece.row, piece.col, direction))
            
        except: 
            pass

        return moves

    def checkDouble(self, piece, colour):
        moves = []
        #piece = self.selected
        row = piece.row
        col = piece.col
        if colour == BLACK or piece.king:
            direction = -1
            moves = self.checkDoubleDirection(direction, row, col, colour, moves, piece)

        if colour == WHITE or piece.king: 
            direction = 1
            moves = self.checkDoubleDirection(direction, row, col, colour, moves, piece)

        return moves
