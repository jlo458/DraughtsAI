import random
from collections import deque

import numpy as np
import tensorflow as ts
from keras._tf_keras.keras.callbacks import TensorBoard
from keras._tf_keras.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam

from draughts import board
from draughts.board import Board
from draughts.consts import WHITE
from draughts.piece import Piece

#from deepQ.tensorBoardMod import ModifiedTensorBoard
#from draughts import board, consts, piece
 

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

    def step(self, action, go):
        self.numMoves += 1 

        reward = 0
        done = False 
        bl, bk, wl, wk = self.board.takeInfo()

        self.board = self.validMoves[action]

        self.episodeNum += 1 

        bla, bka, wla, wka = self.board.takeInfo()

        whiteGo = -1
        if go == 1: 
            whiteGo = 1

        # make reward 
        miniWeight = 1
        reward += ((wla - wl)+(bl - bla))*whiteGo*miniWeight
        reward += ((wka - wk)+(bk - bka))*whiteGo*miniWeight*2

        if self.numMoves >= 60:
            reward -= 10
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

        reward -= 0.1*(self.numMoves)

        return self.vectorise(self.board.board), reward, done, {}

class DQN_Agent: 
    def __init__(self, stateShape = (8,8,1), actionSize = 32, bufferSize = 2000):
        self.stateShape = stateShape
        self.actionSize = actionSize
        self.model = self.buildModel(self.stateShape, self.actionSize)
        self.targetModel = self.buildModel(self.stateShape, self.actionSize)
        self.memory = ReplayBuffer(bufferSize)
        self.epsilon = 1.0
        self.action_map = {}


    def buildModel(self, stateShape, actionSize):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=stateShape))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(actionSize, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model 
    
    def act(self, state, valMoves): 
        if len(valMoves) == 0:
            raise ValueError("No valid moves available")
        
        

        #print(np.random.random())
        if np.random.random() < self.epsilon:
            #print("Hello") 
            action_idx = np.random.randint(len(valMoves))
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            valid_indices = list(self.action_map.keys())  # Indices of valid actions
            try:
                q_values = q_values.flatten()
                valid_q_values = q_values[valid_indices]  

            except: 
                raise ValueError(f"VI: {valid_indices}, QVs; {q_values}")
            
            #print(f"VI {valid_indices}")
            #print(f"VQ {valid_indices}")

            # Choose the action with the highest Q-value
            best_valid_idx = np.argmax(valid_q_values)
            action_idx = valid_indices[best_valid_idx] 
       

        return action_idx  # Convert to actual move
    
    def train(self, batchSize): 
        if self.memory.size() < batchSize: 
            return
        
        batch = self.memory.sample(batchSize)
        states, actions, rewards, nextStates, dones = map(np.array, zip(*batch))

        current_qs = self.model.predict(states)
        next_qs = self.targetModel.predict(nextStates)
        for i in range(batchSize):
            max_future_q = np.max(next_qs[i]) if not dones[i] else 0
            current_qs[i, actions[i]] = rewards[i] + GAMMA * max_future_q

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

'''STATE_SHAPE = (8, 8, 1)  # 8x8 board with a single channel
ACTION_SIZE = 32 

env = draughtEnv()
agent = DQN_Agent()
batchSize = 64

for episode in range(EPISODES): 
    state = env.reset()
    totalReward = 0 
    done = False 

    while not done: 
        validMoves = env.findValidMoves()
        if not validMoves:
            break

        env.action_map = {i: move for i, move in enumerate(validMoves)}

        action = agent.act(state, env.validMoves)
        nextState, reward, done, _ = env.step(action)
        agent.memory.add((state, action, reward, nextState, done))
        agent.train(batchSize)
        state = nextState 
        totalReward += reward

        if env.go == 1: 
            env.go = 2

        else: 
            env.go = 1

    if EPISODES%100 == 0: 
        agent.updateTargetModel()'''


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
        
        return reward, done

def findValidMoves(self): 
        self.validMoves = self.board.possibleMoves()
        return self.validMoves'''
