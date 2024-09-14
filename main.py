# Fix movement bug, may need to switch pieces manually

import pygame
from draughts import board
from draughts.board import Board
from draughts.consts import HEIGHT, WIDTH

FPS = 10
board = Board()

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draughts AI")


def main():
    run = True
    clock = pygame.time.Clock()
    board = Board()

    while run:
        clock.tick(FPS)

        piece = board.select_piece(0, 1)
        #print(piece)
        board.move(piece, 3, 4)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Game logic

        # Board Maker
        # board.buildBoard()
        board.drawAll(window)
        pygame.display.update()

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
