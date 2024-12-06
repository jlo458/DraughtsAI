import random
import time
from collections import deque
from networkx import write_weighted_edgelist
from tqdm import tqdm

import numpy as np
import tensorflow as ts
from keras._tf_keras.keras.callbacks import TensorBoard
from keras._tf_keras.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam

from deepQ.tensorBoardMod import ModifiedTensorBoard
from draughts import board
from draughts.board import Board
from draughts.consts import WHITE
from draughts.piece import Piece

MODEL_NAME = "draughts-dqn"

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_VALUE = 100

GAMMA = 0.99 # Discount factor - higher = future rewards are more important than if it were 0.9

# REWARD ...

EPISODES = 20_000 

epsilon = 1 
EPSILON_DECAY = 0.99975 
MIN_EPSILON = 0.001 

class draughtEnv(): # You will have input board, don't remake it, just swap pieces for 0, 1
    def __init__(self, pos, go) -> None:
        self.board = pos
        self.startGame(pos)

    def reset(self): 
        pos = Board.buildBoard()
        self.startGame(pos)  

    def startGame(self, pos): 
        self.vectorBoard = self.vectorise(pos)
        self.go = 2 
        self.reward = 0
        self.done = False 
        self.episodeNum = 1

    def vectorise(self, board):
        for row in board: 
            for col in range(len(row)): 
                piece = row[col]
                if isinstance(piece, Piece): 
                    if piece.colour == WHITE: 
                        if piece.king: 
                            board[row][col] = 1

                        else: 
                            board[row][col] = 3

                    else: 
                        if piece.king: 
                            board[row][col] = 2

                        else: 
                            board[row][col] = 4

    def step(self, action, go): 
        # reward
        bl, bk, wl, wk = self.board.takeInfo()
        self.board = self.board.move(action)

        self.episodeNum += 1 

        bla, bka, wla, wka = self.board.takeInfo()

        whiteGo = -1
        if go == 1: 
            whiteGo = 1

        # make reward 

        reward = ...
