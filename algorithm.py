# Minimax Algorithm 
# This is where AI will work 

import math
from copy import deepcopy

import pygame

from draughts import board, game

#from draughts.board import Board
from draughts.consts import BLACK, WHITE

#from draughts.game import Game


def miniMax(pos, depth, maxTurn): 
    if depth == 0 or pos.checkWinner():
        return pos.evaluateFunc(), pos # Change if pos is irrevevant 
    
    if maxTurn: 
        maxScore = float("-inf")
        bestMove = None 
        for move in getAllMoves(pos, WHITE): # add game input
            evalScore = miniMax(move, depth-1, False)[0] 
            maxScore = max(evalScore, maxScore)
            if maxScore == evalScore: 
                bestMove = move 

        return maxScore, bestMove
    
    else: 
        minScore = float("inf")
        bestMove = None 
        for move in getAllMoves(pos, BLACK): # add game input
            evalScore, _ = miniMax(move, depth-1, True)
            minScore = min(evalScore, minScore)
            if minScore == evalScore: 
                bestMove = move 

        return minScore, bestMove
    

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
    

