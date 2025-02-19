import time
from copy import deepcopy

import pygame

pygame.init()

from deepQ.altNetwork import DQN_Agent, draughtEnv

#from draughts import board, piece
from draughts.board import Board
from draughts.consts import (
    BLACK,
    BLUE,
    DARK_GREY,
    GREY,
    HEIGHT,
    LIGHT_GREY,
    WHITE,
    WIDTH,
)
from draughts.game import Game
from MiniMax.algorithm import miniMax

#EPISODES = 20_000

# CHANGE MINIMAX SO THAT IT ALWAYS TAKES

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
    #reMove = True
    newRow = move[0]
    newCol = move[1]
    oldRow = piece.row
    oldCol = piece.col
    testBoard.move(piece, newRow, newCol)

    if abs(newRow - oldRow) == 2:  # It's a capture
        middleRow, middleCol = (oldRow + newRow) // 2, (oldCol + newCol) // 2
        testBoard.removePiece(middleRow, middleCol)

        additionalMoves = testBoard.checkDouble(piece, piece.colour)
        for additionalMove in additionalMoves:
            if additionalMove:
                testBoard = simMove(piece, additionalMove, testBoard)

    return testBoard


env = draughtEnv()
agentW = DQN_Agent() # White
agentB = DQN_Agent() # Black
batchSize = 64

#agent = agentW

'''try:
    agentB.model.load_weights("black_9500ver3a_.weights.h5")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")'''

try:
    agentW.model.load_weights("white_6500ver8_.weights.h5")
    print("Loaded existing model weights.")
except Exception as e:
    print(f"Could not load weights: {e}")

def deepQ_move(env, agent, colour): 
    state = env.vectorBoard

    env.validMoves = getAllMoves(env.board, colour)
    if not env.validMoves:
        return None

    agent.action_map = {i: move for i, move in enumerate(env.validMoves)}
    action_idx = agent.act(state, env.validMoves)
    newBoard = env.validMoves[action_idx]


    return newBoard



FPS = 10
board = Board()

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draughts AI")

titleFont = pygame.font.SysFont("Roboto", 150, bold=False)
titleText = titleFont.render("Draughts AI", True, WHITE)

font = pygame.font.SysFont('Helvetica', 45, bold=True)
surfPVP = font.render('PVP', True, 'white')
surfMNM = font.render('MNM', True, 'white')
surfDPQ = font.render('DPQ', True, 'white')

buttonPVP = pygame.Rect(350, 425, 100, 60)
buttonMNM = pygame.Rect(141, 425, 118, 60)
buttonDPQ = pygame.Rect(545, 425, 110, 60)

def draw_gradient_background(win, light, dark):
    height = HEIGHT
    width = WIDTH

    midpoint = height//2

    for y in range(height):
        if y < midpoint:  # Bottom to middle (black to grey)
            fade_color = [
                int(dark[i] + (light[i] - dark[i]) * (y / midpoint))
                for i in range(3)
            ]
        else:  # Middle to top (grey to black)
            fade_color = [
                int(light[i] + (dark[i] - light[i]) * ((y - midpoint) / midpoint))
                for i in range(3)
            ]

        pygame.draw.line(window, fade_color, (0, y), (width, y))

def findMouse(pos): 
    x, y = pos 
    row = y//100 
    col = x//100 
    return row, col

def main():
    run = True
    clock = pygame.time.Clock()
    game = Game(window)

    gameType = None

    #gameType = input("Enter game type: PvP, minimax, deepQ: ")

    # Initial button code

    runButton = True

    while runButton: 
        window.fill(BLACK)
        draw_gradient_background(window, BLUE, BLACK)

        for events in pygame.event.get():
            if events.type == pygame.QUIT: 
                pygame.quit() 
                # check if you need to set run to false

            if events.type == pygame.MOUSEBUTTONDOWN: 
                if buttonPVP.collidepoint(events.pos): 
                    gameType = 'PvP'
                    runButton = False

                elif buttonMNM.collidepoint(events.pos): 
                    gameType = 'minimax'
                    runButton = False

                elif buttonDPQ.collidepoint(events.pos): 
                    gameType = 'deepQ'
                    runButton = False

            a,b = pygame.mouse.get_pos()

            # PvP button
            if buttonPVP.x <= a <= buttonPVP.x + 110 and buttonPVP.y <= b <= buttonPVP.y + 60: 
                pygame.draw.rect(window, LIGHT_GREY, buttonPVP, border_radius=5)

            else: 
                pygame.draw.rect(window, DARK_GREY, buttonPVP, border_radius=5)

            # Minimax button
            if buttonMNM.x <= a <= buttonMNM.x + 110 and buttonMNM.y <= b <= buttonMNM.y + 60: 
                pygame.draw.rect(window, LIGHT_GREY, buttonMNM, border_radius=5)

            else: 
                pygame.draw.rect(window, DARK_GREY, buttonMNM, border_radius=5)

            # Deep Q button
            if buttonDPQ.x <= a <= buttonDPQ.x + 110 and buttonDPQ.y <= b <= buttonDPQ.y + 60: 
                pygame.draw.rect(window, LIGHT_GREY, buttonDPQ, border_radius=5)

            else: 
                pygame.draw.rect(window, DARK_GREY, buttonDPQ, border_radius=5)

            window.blit(titleText, [100, 200])
            
            window.blit(surfPVP, (buttonPVP.x + 5, buttonPVP.y + 5))
            window.blit(surfMNM, (buttonMNM.x + 5, buttonMNM.y + 5))
            window.blit(surfDPQ, (buttonDPQ.x + 5, buttonDPQ.y + 5))

            pygame.display.update()

    # main running loop
    while run:
        clock.tick(FPS)

        whiteMoves = getAllMoves(game.getBoard(), WHITE)
        blackMoves = getAllMoves(game.getBoard(), BLACK)

        winCol = game.getBoard().checkWinner()
        if winCol == BLACK or not whiteMoves:
            print("Black is the winner!")
            run = False 

        if winCol == WHITE or not blackMoves:
            print("White is the winner!")
            run = False

        if game.go == WHITE:
            if gameType == 'deepQ': 
                time.sleep(1)
                env.board = game.board
                newBoard = deepQ_move(env, agentW, WHITE)
                game.minimaxMove(newBoard)

            elif gameType == 'minimax':
                time.sleep(1)
                if (board.blackLeft + board.whiteLeft) < 12:
                    #print("top")
                    score, newBoard = miniMax(game.getBoard(), 6, True, float("-inf"), float("inf")) # game.getBoard not game.board

                else:
                    score, newBoard = miniMax(game.getBoard(), 4, True, float("-inf"), float("inf")) # remember to not have both values   

                #print(score)
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
