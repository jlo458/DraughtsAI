# Minimax Algorithm 
# This is where AI will work 

import math
from copy import deepcopy

import pygame

from draughts.board import Board
from draughts.consts import BLACK, WHITE
from draughts.game import Game


def miniMax(pos, depth, maxTurn): 
    if depth == 0 or pos.checkWinner(): 
        return pos.evaluateFunc(), pos 
    
    if maxTurn: 
        return max(pos, depth-1, maxTurn = False)
    
    else: 
        return min(pos, depth-1, maxTurn = True)
    

