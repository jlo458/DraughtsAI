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

    def takeInfo(self): 
        return self.blackLeft, self.blackKings, self.whiteLeft, self.whiteKings

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
        if isinstance(piece, Piece):
            #print(self.blackLeft, self.whiteLeft, piece.colour)
            if piece.colour == BLACK: 
                if piece.king: 
                    self.blackKings -= 1
                self.blackLeft -= 1

            else: 
                if piece.king: 
                    self.whiteKings -= 1
                self.whiteLeft -= 1 
                #self.checkWinner()

            self.board[row][col] = 0

        #print(self.blackLeft, self.blackKings, self.whiteLeft, self.whiteKings)

    def checkWinner(self): 
        if self.blackLeft == 0: 
            return WHITE, 1  # Make sure to change for the rest of the code

        elif self.whiteLeft == 0: 
            return BLACK, 2
        
        return None

    def evaluateFunc2(self, colour): # Edit as you improve
        w1 = 2
        w2 = 4
        w3 = 1
        #neg = -1
        middlePieces = 0
        winB = float('-inf')
        winW = float('inf')

        theColour = BLACK
        if colour: 
            theColour = WHITE

        # Use this for other evaluation functions - make a bunch
        # 3,3 3,5 4,2 4,4 

        for row in range(3,5): 
            for col in range(3,7,2):
                if row == 4:
                    col = (col-1)
                #print(row, col)
                piece = self.select_piece(row, col)
                
                try:
                    if piece.colour == theColour: 
                        middlePieces += 1

                except: 
                    pass


        pieces = self.whiteLeft - self.blackLeft
        kings = self.whiteKings - self.blackKings

        if not self.checkWinner(): 
            pass

        elif self.checkWinner()[1] == 1: 
            return winW

        elif self.checkWinner()[1] == 2:
            return winB            

        score = w1*(pieces) + w2*(kings) + w3*(middlePieces)
        #print(score, middlePieces, pieces, kings)
        return score

    def getAllPieces(self, colour):
        validPieces = [] 
        for row in self.board:
            for piece in row: 
                if piece != 0 and piece.colour == colour: 
                    validPieces.append(piece)

        return validPieces
    

# Added functions 

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
