from copy import deepcopy
import time
import pygame
import os
#os.environ["OMP_NUM_THREADS"] = '8' - optimisation for when only CPU operating (no GPU)

import re

from draughts.board import Board 
from draughts.piece import Piece
from MiniMax.algorithm import miniMax
from deepQ.altNetwork import DQN_Agent, draughtEnv
from draughts.consts import BLACK, WHITE

# Change updating weights to see if it's better than previous weights 

EPISODES = 10_000
 
def cleanUp(filePattern, lastN):
    modelFiles = [f for f in os.listdir(".") if re.match(filePattern, f)]

    numberedFiles = []
    for file in modelFiles: 
        match = re.search(r"_(\d+)ver11_\.weights\.h5py", file)
        if match:
            numberedFiles.append((int(match.group(1)), file))

    # sorts file
    numberedFiles.sort(key=lambda x: x[0])

    filesToRemove = numberedFiles[:-lastN]
    for _, file_path in filesToRemove:
        os.remove(file_path)
        print(f"Deleted old model: {file_path}")


def getAllMoves(pos, colour):
    if not pos: 
        print("No board entered!")
        return None
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
    newRow = move[0]
    newCol = move[1]
    oldRow = piece.row
    oldCol = piece.col
    testBoard.move(piece, newRow, newCol)
    if abs(newRow - oldRow) == 2:
        testBoard.removePiece((oldRow + newRow) // 2, (oldCol + newCol) // 2)
        moves = testBoard.checkDouble(piece, piece.colour)
        for move in moves:
            if move:  # is not None
                testBoard = simMove(piece, move, testBoard)

    return testBoard


# Change minimax so that it always takes if available
def miniMaxMove(): 
    board = env.board
    if (board.blackLeft + board.whiteLeft) > 12:
        newBoard = miniMax(board, 3, False, float("-inf"), float("inf"))[1] 

    else:
        newBoard = miniMax(board, 5, False, float("-inf"), float("inf"))[1] 

    #print(newBoard)
    return newBoard


env = draughtEnv()
agentW = DQN_Agent()  # White
#agentB = DQN_Agent()  # Black
batchSize = 128

'''try:
    agentB.model.load_weights("black_8000ver5_.weights.h5py")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")'''

try:
    agentW.model.load_weights("white_2500ver11_.weights.h5py", compile=False) # Make sure change
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")

time.sleep(2)

agent = agentW

import csv

with open("rewardsmm.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)  # Activates CSV Writer
    writer.writerow(["Episode", "Reward_White"])  # Makes columns for episode and reward 

    for episode in range(EPISODES+1):
        #agent = agentB
        state = env.reset()
        #totalRewardB = 0
        totalRewardW = 0
        done = False

        while not done:
            if env.go == 1: # Determines whether it's white go (1) or black go (2)
                colour = WHITE
                env.validMoves = getAllMoves(env.board, colour) # Finds all possible moves
                if not env.validMoves:
                    break

                agent.action_map = {i: move for i, move in enumerate(env.validMoves)} # Maps all moves in dictionary

                action_idx = agent.act(state, env.validMoves) # Chooses move 

                nextState, reward, done, _ = env.step(action_idx, env.go) # Determines new board after move

                agent.memory.add((state, action_idx, reward, nextState, done)) # Appends move to memory (Replay Buffer)

                agent.train(batchSize) # Updates training model 

                state = nextState # Changes state

            else:
                colour = BLACK
                newBoard = miniMaxMove()
                if not newBoard: 
                    done = True

                else:
                    done = env.miniMaxStep(newBoard, env.go)[1]

            if env.go == 1:
                totalRewardW += reward
                env.go = 2
                #agent = agentB

            else:
                #totalRewardB += reward
                env.go = 1
                #agent = agentW

        print(episode)
        if episode % 50 == 0:
            agent.updateTargetModel()
            #print(f"Total Reward for Black: {totalRewardB}, Episode: {episode}")
            print(f"Total Reward for White: {totalRewardW}, Episode: {episode+2500}")
            writer.writerow([episode+2500, totalRewardW])

        if episode % 500 == 0:
            agentW.model.save_weights(f"white_{episode+2500}ver11_.weights.h5py")
            #agentB.model.save_weights(f"black_{episode}ver7_.weights.h5")

            #writer.writerow([episode, totalRewardB, totalRewardW])

            cleanUp(r"white_\d+ver11_\.weights\.h5py", 3)  # change this
            #cleanUp(r"black_\d+ver11_\.weights\.h5", 3)

#agentB.model.save_weights("black.weights.h5")
agentW.model.save_weights("white.weights.h5py")
print("Training Complete!")








