import pygame

from draughts import board, piece
from draughts.board import Board
from draughts.consts import HEIGHT, WIDTH
from draughts.game import Game

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos() 
                row, col = findMouse(pos)
                game.check(row, col) 
                
                
                #board.move(piece, 4, 3)


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
