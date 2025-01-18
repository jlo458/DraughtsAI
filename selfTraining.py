# This is the code for training AI against itself

from copy import deepcopy

import time

import pygame

import os
import re

from draughts.board import Board 
from draughts.piece import Piece
from deepQ.altNetwork import DQN_Agent, draughtEnv
from draughts.consts import BLACK, WHITE


EPISODES = 15_000
 
def cleanUp(filePattern, lastN):
    modelFiles = [f for f in os.listdir(".") if re.match(filePattern, f)]

    numberedFiles = []
    for file in modelFiles:
        match = re.search(r"_(\d+)ver6_\.weights\.h5", file)
        if match:
            numberedFiles.append((int(match.group(1)), file))

    # sorts file
    numberedFiles.sort(key=lambda x: x[0])

    filesToRemove = numberedFiles[:-lastN]
    for _, file_path in filesToRemove:
        os.remove(file_path)
        print(f"Deleted old model: {file_path}")


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
    # reMove = True
    newRow = move[0]
    newCol = move[1]
    oldRow = piece.row
    oldCol = piece.col
    testBoard.move(piece, newRow, newCol)
    if newRow == oldRow - 2 or newRow == oldRow + 2:
        testBoard.removePiece((oldRow + newRow) // 2, (oldCol + newCol) // 2)
        moves = testBoard.checkDouble(piece, piece.colour)
        for move in moves:
            if move is not None:
                testBoard = simMove(piece, move, testBoard)

    return testBoard


env = draughtEnv()
agentW = DQN_Agent()  # White
agentB = DQN_Agent()  # Black
batchSize = 64

'''try:
    agentB.model.load_weights("black_8000ver5_.weights.h5")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")'''

try:
    agentW.model.load_weights("white_11500ver5_.weights.h5")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")

time.sleep(2)

import csv

with open("rewards5mm.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Episode", "Reward_Black", "Reward_White"])

    for episode in range(EPISODES+1):
        agent = agentB
        state = env.reset()
        totalRewardB = 0
        totalRewardW = 0
        done = False

        while not done:
            if env.go == 1:
                colour = WHITE

            else:
                colour = BLACK

            env.validMoves = getAllMoves(env.board, colour)
            if not env.validMoves:
                break

            agent.action_map = {i: move for i, move in enumerate(env.validMoves)}

            action_idx = agent.act(state, env.validMoves)
            nextState, reward, done, _ = env.step(action_idx, env.go)
            agent.memory.add((state, action_idx, reward, nextState, done))
            agent.train(batchSize)
            state = nextState

            if env.go == 1:
                totalRewardW += reward
                env.go = 2
                agent = agentB

            else:
                totalRewardB += reward
                env.go = 1
                agent = agentW

        print(episode)
        if episode % 50 == 0:
            agent.updateTargetModel()
            print(f"Total Reward for Black: {totalRewardB}, Episode: {episode}")
            print(f"Total Reward for White: {totalRewardW}, Episode: {episode}")
            writer.writerow([episode, totalRewardB, totalRewardW])

        if episode % 500 == 0:
            agentW.model.save_weights(f"white_{episode}ver7_.weights.h5")
            agentB.model.save_weights(f"black_{episode}ver7_.weights.h5")

            #writer.writerow([episode, totalRewardB, totalRewardW])

            cleanUp(r"white_\d+ver7_\.weights\.h5", 3)  # change this
            cleanUp(r"black_\d+ver7_\.weights\.h5", 3)

agentB.model.save_weights("black.weights.h5")
agentW.model.save_weights("white.weights.h5")
print("Training Complete!")
