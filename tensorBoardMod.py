'''import tensorflow as tf
from keras._tf_keras.keras.callbacks import TensorBoard


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)'''

import pygame

from .consts import BLACK, BLUE, WHITE
from .piece import Piece


class Board:
    def __init__(self) -> None:
        self.board = []
        self.chosenPiece = None
        self.blackLeft = self.whiteLeft = 12
        self.blackKings = self.whiteKings = 0
        self.buildBoard()

    def drawBoard(self, window):
        window.fill(WHITE)
        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 1:
                    pygame.draw.rect(window, BLUE, [100 * x, 100 * y, 100, 100])

        pygame.display.flip()

    def buildBoard(self):
        for row in range(8):
            self.board.append([])
            for col in range(8):
                if (row + col) % 2 == 1:
                    if row < 3:
                        self.board[row].append(Piece(row, col, WHITE))

                    elif row > 4:
                        self.board[row].append(Piece(row, col, BLACK))

                    else:
                        self.board[row].append(0)

                else:
                    self.board[row].append(0)

    def select_piece(self, row, col):
        return self.board[row][col]

    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)

        if row == 0 or row == 7:
            piece.makeKing()
            if piece.colour == WHITE:
                self.whiteKings += 1

            else:
                self.blackKings += 1

    def drawAll(self, window):
        self.drawBoard(window)
        for row in range(8):
            for col in range(8):
                object = self.board[row][col]
                if object != 0:
                    object.drawPiece(window)

    def removePiece(self, row, col):
        piece = self.board[row][col]
        if piece.colour == BLACK:
            self.blackLeft -= 1
            self.checkWinner()

        else:
            self.whiteLeft -= 1
            self.checkWinner()

        self.board[row][col] = 0

    def checkWinner(self):
        if self.blackLeft == 0:
            print("White Wins!")
            return True

        elif self.whiteLeft == 0:
            print("Black Wins!")
            return True

        return False

    def evaluateFunc(self):  # Edit as you improve
        w1 = 2
        w2 = 1
        score = w1 * (self.whiteLeft - self.blackLeft) + w2 * (self.whiteKings - self.blackKings)

        return score

    def getAllPieces(self, colour):
        validPieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.colour == colour:
                    validPieces.append(piece)

        return validPieces

    def possibleMoves(self, piece):
        moves = []
        row = piece.row
        col = piece.col
        colour = piece.colour

        if colour == BLACK or piece.king:
            moves.append(self.checkRight(row, col, colour, -1))
            moves.append(self.checkLeft(row, col, colour, -1))

        if colour == WHITE or piece.king:
            moves.append(self.checkRight(row, col, colour, 1))
            moves.append(self.checkLeft(row, col, colour, 1))

        return moves

    def checkRight(self, row, col, colour, direction):
        if col == 7 or (row + direction) > 7 or (row + direction) < 0:
            return None

        rightSpace = self.select_piece(row + direction, col + 1)
        if rightSpace == 0:
            return row + direction, col + 1

        elif rightSpace.colour == colour:
            return None

        else:
            return self._takeRight(row, col, direction)

    def _takeRight(self, row, col, direction):
        if col >= 6 or row + (2 * direction) > 7 or row + (2 * direction) < 0:
            return None

        rightSpace = self.select_piece(row + (2 * direction), col + 2)
        if rightSpace == 0:
            return row + (2 * direction), col + 2

        return None

    def checkLeft(self, row, col, colour, direction):
        if col == 0 or (row + direction) > 7 or (row + direction) < 0:
            return None

        leftSpace = self.select_piece(row + direction, col - 1)
        if leftSpace == 0:
            return row + direction, col - 1

        elif leftSpace.colour == colour:
            return None

        else:
            return self._takeLeft(row, col, direction)

    def _takeLeft(self, row, col, direction):
        if col <= 1 or row + (2 * direction) > 7 or row + (2 * direction) < 0:
            return None
        leftSpace = self.select_piece(row + (2 * direction), col - 2)
        if leftSpace == 0:
            return row + (2 * direction), col - 2

        return None

    def checkDoubleDirection(self, direction, row, col, colour, moves, piece):
        try:
            rightSpace = self.select_piece(row + direction, col + 1)
            if rightSpace.colour != colour:
                moves.append(self._takeRight(piece.row, piece.col, direction))

        except:
            pass

        try:
            leftSpace = self.select_piece(row + direction, col - 1)
            if leftSpace.colour != colour:
                moves.append(self._takeLeft(piece.row, piece.col, direction))

        except:
            pass

        return moves

    def checkDouble(self, piece, colour):
        moves = []
        # piece = self.selected
        row = piece.row
        col = piece.col
        if colour == BLACK or piece.king:
            direction = -1
            moves = self.checkDoubleDirection(direction, row, col, colour, moves, piece)

        if colour == WHITE or piece.king:
            direction = 1
            moves = self.checkDoubleDirection(direction, row, col, colour, moves, piece)

        return moves

# old altNetwork 

import random
from collections import deque

import numpy as np
import tensorflow as tf

from keras._tf_keras.keras.layers import Conv2D, Dense, Flatten
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.optimizers import Adam

from draughts.board import Board
from draughts.consts import WHITE
from draughts.piece import Piece

GAMMA = 0.99  # Discount factor - higher = future rewards are more important than if it were 0.9

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

class draughtEnv():  # You will have input board, don't remake it, just swap pieces for 0, 1
    def __init__(self, pos=None) -> None:
        board = Board()
        self.board = board
        # print(board.board)
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

    def vectorise(self, board):
        grid = np.zeros((8, 8), dtype=int)
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
        reward += ((wla - wl) + (bl - bla)) * whiteGo * miniWeight
        reward += ((wka - wk) + (bk - bka)) * whiteGo * miniWeight * 2

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

        return self.vectorise(self.board.board), reward, done, {}


class DQN_Agent:
    def __init__(self, stateShape=(8, 8, 1), actionSize=32, bufferSize=2000):
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

        # print(np.random.random())
        if np.random.random() < self.epsilon:
            # print("Hello")
            action_idx = np.random.randint(len(valMoves))
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            valid_indices = list(self.action_map.keys())  # Indices of valid actions
            try:
                q_values = q_values.flatten()
                valid_q_values = q_values[valid_indices]

            except:
                raise ValueError(f"VI: {valid_indices}, QVs; {q_values}")

            best_valid_idx = np.argmax(valid_q_values)
            action_idx = valid_indices[best_valid_idx]

        return action_idx

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

