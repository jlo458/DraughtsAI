imporimport random
import time
from collections import deque
from tqdm import tqdm
import numpy

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
        done = False 
        bl, bk, wl, wk = self.board.takeInfo()
        self.board = self.board.move(action)

        self.episodeNum += 1 

        bla, bka, wla, wka = self.board.takeInfo()

        whiteGo = -1
        if go == 1: 
            whiteGo = 1

        # make reward 
        miniWeight = 1
        reward = ((wla - wl)+(bl - bla) + (wka - wk)+(bk - bka))*whiteGo*miniWeight

        if self.board.checkWinner(): 
            done = True

        return reward, done

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
    
    def act(self, state): 
        if np.random.random() < self.epsilon: 
            pass 
            # Random Action 
            # Return np.random.randint(self.action_size)

        q_values = self.model.predict(state[np.newaxis])
        return numpy.argmax(q_values)
    
    def train(self, batchSize): 
        if self.memorySize < batchSize: 
            return
        
        batch = self.memory.sample(batchSize)
        states, actions, rewards, nextStates, dones = map(np.array, zip(*batch))

        target_qs = self.model.predict(states)
        next_qs = self.target_model.predict(nextStates)
        for i in range(batchSize):
            target_qs[i, actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * np.max(next_qs[i])

        self.model.fit(states, target_qs, verbose=0)

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
        action = agent.act(state)
        nextState, reward, done, _ = env.step(action)
        agent.memory.add((state, action, reward, nextState, done))
        agent.train(batchSize)
        state = nextState 
        totalReward = reward

agent.updateTargetModel()
print(f"Episode {episode}, Total Reward: {totalReward}")
t random
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
