# Minimax Algorithm 
# This is where AI will work 

import math
from copy import deepcopy

import pygame

from draughts import board, game
from draughts.board import Board
from draughts.consts import BLACK, WHITE
from draughts.game import Game


def miniMax(pos, depth, maxTurn): 
    if depth == 0 or pos.checkWinner(): 
        return pos.evaluateFunc(), pos 
    
    if maxTurn: 
        maxScore = float("-inf")
        bestMove = None 
        for move in getAllMoves(pos, WHITE): # add game input
            evalScore = miniMax(move, depth-1, False) 
            maxScore = max(evalScore, maxScore)
            if maxScore == evalScore: 
                bestMove = move 
    
    else: 
        minScore = float("inf")
        bestMove = None 
        for move in getAllMoves(pos, BLACK): # add game input
            evalScore = miniMax(move, depth-1, True) 
            minScore = min(evalScore, maxScore)
            if minScore == evalScore: 
                bestMove = move 
    
    

def getAllMoves(pos, col): 
    moves = []
    for piece in board.getAllPieces(col): 
        validMoves = game.getValidMoves(piece)
        for move in validMoves:
            testBoard = deepcopy(pos)
            moves.append([simMove(piece, move, testBoard), piece])

def simMove(self, piece, move, testBoard): 
    testBoard.move(piece, move[0], move[1])
    return testBoard
    

