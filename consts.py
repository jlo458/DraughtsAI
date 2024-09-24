import pygame

# Dimensions
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# RGB Colours
RED = (255, 0, 0)
BLUE = (10, 150, 140)
WHITE = (255, 255, 255)
BLACK = (50, 50, 50)
GREY = (125, 125, 125)

# King sign
CROWN = pygame.transform.scale(pygame.image.load("draughts/crown.png"), (45, 25))
