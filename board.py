import pygame

# from draughts import piece
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
        #print(row, col)
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
                    # print("Hello")
                    object.drawPiece(window)

    def removePiece(self, row, col):
        piece = self.board[row][col]
        if piece.colour == BLACK: 
            self.blackLeft -= 1
            if self.blackLeft == 0: 
                print("White wins!") 

        else: 
            self.whiteLeft -= 1 
            if self.whiteLeft: 
                print("Black wins!")

        
        self.board[row][col] = 0
