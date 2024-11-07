# Minimax Algorithm 
# This is where AI will work 

import math
from copy import deepcopy

import pygame

from draughts import board, game
from draughts.consts import BLACK, WHITE


def miniMax(pos, depth, maxTurn, alpha, beta): 
    if depth == 0 or pos.checkWinner():
        return pos.evaluateFunc(), pos # Change if pos is irrevevant 
    
    if maxTurn: 
        maxEval = float("-inf")
        bestMove = None 
        for move in getAllMoves(pos, WHITE): 
            evalScore = miniMax(move, depth-1, False, alpha, beta)[0] 
            maxEval = max(evalScore, maxEval)

            alpha = max(maxEval, alpha) 
            if alpha >= beta:
                #print(f"Alphabeta pruning depth {depth}") 
                break

            if maxEval == evalScore: 
                bestMove = move 

        return maxEval, bestMove
    
    else: 
        minEval = float("inf")
        bestMove = None 
        for move in getAllMoves(pos, BLACK): # add game input
            evalScore, _ = miniMax(move, depth-1, True, alpha, beta)
            minEval = min(evalScore, minEval)

            beta = min(minEval, beta) 
            if alpha >= beta: 
                #print(f"Alphabeta pruning depth {depth}") 
                break

            if minEval == evalScore: 
                bestMove = move 

        return minEval, bestMove
    

def getAllMoves(pos, colour): 
    positions = []
    for piece in pos.getAllPieces(colour): 
        validMoves = pos.possibleMoves(piece)
        for move in validMoves:
            if move == None: 
                continue
            testBoard = deepcopy(pos)
            testPiece = testBoard.select_piece(piece.row, piece.col)
            positions.append(simMove(testPiece, move, testBoard))

    return positions

def simMove(piece, move, testBoard):
    #reMove = True
    newRow = move[0]
    newCol = move[1]
    oldRow = piece.row
    oldCol = piece.col
    testBoard.move(piece, newRow, newCol)
    if newRow == oldRow-2 or newRow == oldRow+2:
        testBoard.removePiece((oldRow+newRow)//2, (oldCol+newCol)//2)
        moves = testBoard.checkDouble(piece, piece.colour)
        for move in moves: 
            if move is not None: 
                testBoard = simMove(piece, move, testBoard)

    return testBoard
    

'''
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
                self.changeTurn()'''
    

