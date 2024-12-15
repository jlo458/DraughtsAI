import random
import time
from collections import deque
from pickle import TRUE
from tabnanny import verbose

import numpy
import numpy as np
import tensorflow as ts
from keras._tf_keras.keras.callbacks import TensorBoard
from keras._tf_keras.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam
from tqdm import tqdm

from deepQ.tensorBoardMod import ModifiedTensorBoard
from draughts import board
from draughts.board import Board
from draughts.consts import WHITE
from draughts.piece import Piece

MODEL_NAME = "draughts-dqn"

'''REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_VALUE = 100'''

GAMMA = 0.99 # Discount factor - higher = future rewards are more important than if it were 0.9

# REWARD ...

EPISODES = 20_000 

#epsilon = 1 
EPSILON_DECAY = 0.99975 
MIN_EPSILON = 0.001 

class draughtEnv(): # You will have input board, don't remake it, just swap pieces for 0, 1
    def __init__(self, pos=None) -> None:
        self.board = pos or Board.buildBoard()
        self.startGame(pos)

    def reset(self): 
        pos = Board.buildBoard()
        self.startGame(pos)  
        return self.vectorBoard  # check

    def startGame(self, pos): 
        self.vectorBoard = self.vectorise(pos)
        self.validMoves = None
        self.action_map = None
        self.go = 2 
        self.reward = 0
        self.done = False 
        self.episodeNum = 1

    def findValidMoves(self, board): 
        self.validMoves = self.board.possibleMoves()
        return self.validMoves

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

        return board

    def step(self, action, go):
        done = False 
        bl, bk, wl, wk = self.board.takeInfo()

        theMove = self.validMoves[action]
        self.board = self.board.move(theMove)

        self.episodeNum += 1 

        bla, bka, wla, wka = self.board.takeInfo()

        whiteGo = -1
        if go == 1: 
            whiteGo = 1

        # make reward 
        miniWeight = 1
        reward += ((wla - wl)+(bl - bla))*whiteGo*miniWeight
        reward += ((wka - wk)+(bk - bka))*whiteGo*miniWeight*2

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

        return self.vectorise(self.board), reward, done

class DQN_Agent: 
    def __init__(self, stateShape, actionSize, bufferSize = 2000):
        self.stateShape = stateShape
        self.actionSize = actionSize
        self.model = self.buildModel(stateShape, actionSize)
        self.targetModel = self.buildModel((stateShape, actionSize))
        self.memory = ReplayBuffer(bufferSize)
        self.epsilon = 1.0


    def buildModel(stateShape, actionSize):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=stateShape))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(actionSize, activation='linear'))
        model.compile(optmiser=Adam(learning_rate=0.001), loss='mse')
        return model 
    
    def act(self, state, valMoves): 
        if np.random.random() < self.epsilon: 
            action_idx = np.random.randint(len(valMoves))
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            action_idx = np.argmax(q_values[:len(valMoves)])  # Restrict to valid moves
        return self.action_map[action_idx]  # Convert to actual move
    
    def train(self, batchSize): 
        if self.memorySize < batchSize: 
            return
        
        batch = self.memory.sample(batchSize)
        states, actions, rewards, nextStates, dones = map(np.array, zip(*batch))

        current_qs = self.model.predict(states)
        next_qs = self.target_model.predict(nextStates)
        for i in range(batchSize):
            max_future_q = np.max(next_qs[i]) if not dones[i] else 0
            current_qs[i, actions[i]] = rewards[i] + self.gamma * max_future_q

        self.model.fit(states, current_qs, verbose=0)

        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)


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

# sort this out
env = draughtEnv()
agent = DQN_Agent()
batchSize = 64

for episode in range(EPISODES): 
    state = env.reset()
    totalReward = 0 
    done = False 

    while not done: 
        validMoves = env.findValidMoves()
        env.action_map = {i: move for i, move in enumerate(validMoves)}

        action = agent.act(state, env.validMoves)
        nextState, reward, done, _ = env.step(action)
        agent.memory.add((state, action, reward, nextState, done))
        agent.train(batchSize)
        state = nextState 
        totalReward = reward

agent.updateTargetModel()
print(f"Episode {episode}, Total Reward: {totalReward}")

# Proper reward system 

'''
class draughtEnv:
    ...
    def step(self, action):
        ...
        # Compute changes in piece counts
        reward = 0
        bla, bka, wla, wka = self.board.takeInfo()

        # Reward based on capturing pieces
        reward += (wl - wla) - (bl - bla)  # Difference in lost pieces
        reward += 2 * ((wk - wka) - (bk - bka))  # Higher weight for kings

        # Reward for winning or penalty for losing
        if self.board.checkWinner() == 1:  # White wins
            reward += 100
            done = True
        elif self.board.checkWinner() == -1:  # Black wins
            reward -= 100
            done = True
        else:
            done = False

        if len(self.validMoves) == 0:
            reward -= 50
        
        return reward, done'''
