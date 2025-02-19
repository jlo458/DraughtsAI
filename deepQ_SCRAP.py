import time
from copy import deepcopy

import pygame

from deepQ.altNetwork import DQN_Agent, draughtEnv

#from deepQ.network import EPISODES
from draughts import board, piece
from draughts.board import Board
from draughts.consts import BLACK, HEIGHT, WHITE, WIDTH
from draughts.game import Game
from MiniMax.algorithm import miniMax

EPISODES = 20_000
MODEL_NAME = "draughts-dqn"

GAMMA = 0.99 # Discount factor - higher = future rewards are more important than if it were 0.9

#epsilon = 1 
EPSILON_DECAY = 0.99975 
MIN_EPSILON = 0.001 

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
    #reMove = True
    newRow = move[0]
    newCol = move[1]
    oldRow = piece.row
    oldCol = piece.col
    testBoard.move(piece, newRow, newCol)
    if newRow == oldRow-2 or newRow == oldRow+2:
        testBoard.removePiece((oldRow+newRow)//2, (oldCol+newCol)//2)
        moves = testBoard.checkDouble(piece, piece.colour)
        for move in moves: 
            if move is not None: 
                testBoard = simMove(piece, move, testBoard)

    return testBoard

env = draughtEnv()
agentW = DQN_Agent() # White
agentB = DQN_Agent() # Black
batchSize = 64

try:
    agentB.model.load_weights("black.weights.h5")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")

try:
    agentW.model.load_weights("white.weights.h5")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")

for episode in range(EPISODES): 
    agent = agentB
    state = env.reset()
    totalReward = 0 
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
        totalReward += reward

        if env.go == 1: 
            env.go = 2
            agent = agentB

        else: 
            env.go = 1
            agent = agentW

    print(episode)
    if episode%50 == 0: 
        agent.updateTargetModel()
        print(f"Total Reward: {totalReward}, Episode: {episode}")

    if episode%100 == 0: 
        agentW.model.save_weights(f"white_{episode}_.weights.h5")
        agentB.model.save_weights(f"black_{episode}_.weights.h5")


agentB.model.save_weights("black.weights.h5")
agentW.model.save_weights("white.weights.h5")
print("Training Complete!")


'''

FPS = 10
board = Board()

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draughts AI")

def findMouse(pos): 
    x, y = pos 
    row = y//100 
    col = x//100 
    return row, col

def main():
    run = True
    clock = pygame.time.Clock()
    game = Game(window)

    while run:
        clock.tick(FPS)

        winCol = game.getBoard().checkWinner()
        if winCol == BLACK:
            print("Black is the winner!")
            run = False 

        if winCol == WHITE:
            print("White is the winner!")
            run = False




        if game.go == WHITE:
            #newBoard = miniMax(game.getBoard(), 3, True, float("-inf"), float("inf"))[1] # game.getBoard not game.board 
            newBoard = deepQalg(game.getBoard())

            game.minimaxMove(newBoard)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos() 
                row, col = findMouse(pos)
                game.select(row, col) 
                
        game.update()


    pygame.quit()


main()


'''

# Code for words on right hand side
# Can be used for leadreboard etc
# Just need to adjust board WIDTH to 1000
"""
pygame.font.init()

titleFont = pygame.font.SysFont("Calibri", 25, True, False)
        text = titleFont.render("Draughts AI", True, WHITE)
        window.blit(text, [840, 100])
"""
