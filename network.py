import random
import time
from collections import deque
from tqdm import tqdm

import numpy as np
import tensorflow as ts
from keras._tf_keras.keras.callbacks import TensorBoard
from keras._tf_keras.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam

from deepQ.tensorBoardMod import ModifiedTensorBoard

MODEL_NAME = "draughts-dqn"
log_dir = "logs/{}-{}".format(MODEL_NAME, int(time.time()))

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

# Maybe incorporate stat setting if necessary

# In progress
# fix
class draughtEnv(): # You will have input board, don't remake it, just swap pieces for 0, 1
    def __init__(self, pos) -> None:
        self.board = self.changeToNumBoard(pos)
        self.chosenPiece = None

    def reset(): 
        pass  

class DQNAgent:
    def __init__(self) -> None: 
        # Main Model - trained every step
        self.model = self.makeModel() 

        # Target model - against what main model is predicted
        self.targetModel = self.makeModel() 
        self.targetModel.set_weights(self.model.get_weights())

        self.replayMemory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=log_dir, histogram_freq=1)

        self.targetUpdateCounter = 0


        
    def makeModel(self):
        model = Sequential() # Type of network

        model.add(Conv2D(32, (3, 3), input_shape=(8, 8, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten()) # Changes into 1D vector format for dense()

        model.add(Dense(64, activation='linear')) # Makes decision

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear')) # Output

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model
    
    def updateReplayMemory(self, transition): 
        self.replayMemory.append(transition)

    def getQs(self, state, step): 
        return self.modelPredict(np.array(state).reshape(-1, *state.shape)/255)[0]
    

    def train(self, terminalState, step): 
        if len(self.replayMemory) < MIN_REPLAY_MEMORY_SIZE: 
            return 
        
        minibatch = random.sample(self.replayMemory, MINIBATCH_SIZE)
        currentStates = np.array([transition[0] for transition in minibatch])/255
        currentQsList = self.model.predict(currentStates)

        newCurrentStates = np.array([transition[3] for transition in minibatch])/255
        futureQsList = self.targetModel.predict(newCurrentStates)

        X = []
        Y = []

        for index, (currentState, action, reward, newCurrentState, done) in enumerate(minibatch): 
            if not done: 
                maxFutureQ = np.max(futureQsList[index])
                newQ = reward + GAMMA * maxFutureQ

            else: 
                newQ = reward

            currentQs = currentQsList[index]
            currentQs[action] = newQ 

            X.append(currentState)
            Y.append(currentQs) 

        self.model.fit(np.array(X)/255, np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminalState else None)
        
        if terminalState: 
            self.targetUpdateCounter += 1

        if self.targetUpdateCounter > UPDATE_TARGET_VALUE: 
            self.targetModel.set_weights(self.model.get_weights())
            self.targetUpdateCounter = 0

# Not part of class 

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episode"): 
    agent.tensorboard.step = episode 

    episodeReward = 0 
    step = 1
    currentState = env.reset()

    done = False 

    while not done: 
        if np.random.random() > epsilon: 
            action = np.argmax(agent.getQs(currentState))

        else: 
            pass 
            # random move 

        newState, reward, done = env.step(action) 

        episodeReward += reward 

        # show preview stuff 

        agent.updateReplayMemory(currentState, action, reward, newState, done)
        agent.train(done, step)

        currentState = newState
        step += 1

        # ep_rewards stuff
