# I've had to modify this code to work with a GPU (CUDA 11.8, cuDNN 8.9), so I had to change the tensorflow version to 2.10 
# I also found it easier to operate on conda when utilising a GPU

import random
from collections import deque

import os
#os.environ["OMP_NUM_THREADS"] = '4' - optimisation for just CPU

import numpy as np
import tensorflow as tf

# Various optimiastion techniques for operating with a GPU
from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(tf.__version__)

from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

# Making sure GPU is active 
print("Built with CUDA:", tf.test.is_built_with_cuda())  
print("GPUs available: ", tf.config.list_physical_devices('GPU'))

from draughts.board import Board
from draughts.consts import WHITE
from draughts.piece import Piece

GAMMA = 0.99 # Discount factor - higher = future rewards are more important than if it were 0.9
EPISODES = 10_000 

EPSILON_DECAY = 0.99985 
MIN_EPSILON = 0.1 

class draughtEnv(): # You will have input board, don't remake it, just swap pieces for 0, 1
    def __init__(self, pos=None) -> None:
        board = Board()
        self.board = board
        #print(board.board)
        self.startGame(self.board.board)

    def reset(self): 
        board = Board()
        self.board = board
        self.startGame(self.board.board)  
        return self.vectorBoard  # check

    def startGame(self, pos): 
        self.vectorBoard = self.vectorise(pos)
        self.validMoves = None
        self.action_map = None
        self.go = 2 
        self.reward = 0
        self.done = False 
        self.episodeNum = 1
        self.numMoves = 0

    def vectorise(self, board):
        grid = np.zeros((8,8), dtype=int)
        for row in range(8): 
            for col in range(8): 
                piece = board[row][col]
                if isinstance(piece, Piece): 
                    if piece.colour == WHITE: 
                        if piece.king: 
                            grid[row][col] = 3

                        else: 
                            grid[row][col] = 1

                    else: 
                        if piece.king: 
                            grid[row][col] = 4

                        else: 
                            grid[row][col] = 2

        return grid
    

    def miniMaxStep(self, action, go): 
        done = False
        self.numMoves += 1
        self.board = action 
        if self.board.checkWinner():
            if self.board.checkWinner()[1] == go:
                done = True

            if self.board.checkWinner()[1] != go:
                done = True

        return self.vectorise(self.board.board), done, {}

    def gamePhase(self, board): 
        if (board.blackLeft + board.whiteLeft) > 16: 
            return 5 
        
        elif (board.blackLeft + board.whiteLeft) > 10:
            return 15 
        
        else: 
            return 30

    def step(self, action, go):
        self.numMoves += 1 

        reward = 0
        done = False 
        bl, bk, wl, wk = self.board.takeInfo() # Info of current board

        self.board = self.validMoves[action]

        self.episodeNum += 1 

        bla, bka, wla, wka = self.board.takeInfo() # Info after move is made 

        whiteGo = -1
        if go == 1: 
            whiteGo = 1

        miniWeight = self.gamePhase(self.board)*5
        reward += ((wla - wl)+(bl - bla))*whiteGo*miniWeight
        reward += ((wka - wk)+(bk - bka))*whiteGo*miniWeight*2.5

        if self.numMoves >= 50:
            reward -= 5
            done = True

        if self.board.checkWinner():
            if self.board.checkWinner()[1] == go:
                reward += 1000 
                done = True

            if self.board.checkWinner()[1] != go:
                reward -= 1000 
                done = True

        if len(self.validMoves) == 0:
            reward -= 900
            done = True

        reward -= 0.2*(self.numMoves)

        return self.vectorise(self.board.board), reward, done, {}

class DQN_Agent: 
    def __init__(self, stateShape = (8,8,1), actionSize = 32, bufferSize = 2000):
        self.stateShape = stateShape
        self.actionSize = actionSize
        self.model = self.buildModel(self.stateShape, self.actionSize)
        self.targetModel = self.buildModel(self.stateShape, self.actionSize)
        self.memory = ReplayBuffer(bufferSize)
        self.epsilon = 0.8
        self.action_map = {}

    # Previous model
    '''def buildModel(self, stateShape, actionSize):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=stateShape))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(actionSize, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model '''
    

    def buildModel(self, stateShape, actionSize):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=stateShape))
        model.add(Conv2D(64, (3, 3), activation='relu'))  # Added another Conv layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))  # Added another Dense layer
        model.add(Dense(actionSize, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state, valMoves): 
        if len(valMoves) == 0:
            raise ValueError("No valid moves available")
        
        if np.random.random() < self.epsilon: 
            action_idx = np.random.randint(len(valMoves))

        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            valid_indices = list(self.action_map.keys())  # Indices of valid actions
            try:
                q_values = q_values.flatten()
                valid_q_values = q_values[valid_indices]  

            except: 
                raise ValueError(f"VI: {valid_indices}, QVs; {q_values}")

            # Choose the action with the highest Q-value
            best_valid_idx = np.argmax(valid_q_values)
            action_idx = valid_indices[best_valid_idx] 
       
        return action_idx  # Convert to actual move
    
    def train(self, batchSize): 
        if self.memory.size() < batchSize: # Makes sure there are enough experiences for a batch
            return
        
        batch = self.memory.sample(batchSize) # Takes a sample of experiences

        states, actions, rewards, nextStates, dones = map(np.array, zip(*batch)) # Splits experiences into seperate parts

        current_qs = self.model.predict(states) # Predicts current q-values

        next_qs = self.targetModel.predict(nextStates) # Predicts q-values of future positions

        for i in range(batchSize):
            max_future_q = np.max(next_qs[i]) if not dones[i] else 0  # Finds most optimal future position
            current_qs[i, actions[i]] = rewards[i] + GAMMA * max_future_q  # Bellman Equation

        self.model.fit(states, current_qs, verbose=0) # Fits model with updates q-values

        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY) # Decreases epsilon


    def updateTargetModel(self): 
        self.targetModel.set_weights(self.model.get_weights())

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
