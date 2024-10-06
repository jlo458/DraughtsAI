import pygame

from draughts import board, piece
from draughts.board import Board
from draughts.consts import HEIGHT, WHITE, WIDTH
from draughts.game import Game
from MiniMax.algorithm import miniMax

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

        if game.go == WHITE:
            newBoard = miniMax(game.getBoard(), 4, True)[1] # game.getBoard not game.board
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


# Code for words on right hand side
# Can be used for leadreboard etc
# Just need to adjust board WIDTH to 1000
"""
pygame.font.init()

titleFont = pygame.font.SysFont("Calibri", 25, True, False)
        text = titleFont.render("Draughts AI", True, WHITE)
        window.blit(text, [840, 100])
"""
