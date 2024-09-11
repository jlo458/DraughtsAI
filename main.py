import pygame
from draughts.board import Board
from draughts.consts import HEIGHT, WIDTH

FPS = 10

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draughts AI")


def main():
    run = True
    clock = pygame.time.Clock()
    board = Board()

    while run:
        clock.tick(FPS)

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
in workings of game

import pygame
from draughts.consts import BLACK, BLUE, HEIGHT, WIDTH

FPS = 60

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draughts AI")


def main():
    run = True
    clock = pygame.time.Clock()

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Game logic

        # Board Maker
        window.fill(BLACK)
        for x in range(8):
            for y in range(8):
                if (x + y) % 2 == 0:
                    pygame.draw.rect(window, BLUE, [100 * x, 100 * y, 100, 100])

        pygame.display.flip()

    pygame.quit()


main()


# Code for words on right hand side
# Can be used for leadreboard etc 
# Just need to adjust board WIDTH to 1000
'''
pygame.font.init()

titleFont = pygame.font.SysFont("Calibri", 25, True, False)
        text = titleFont.render("Draughts AI", True, WHITE)
        window.blit(text, [840, 100])
'''
