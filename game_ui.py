import pygame

from engine import Engine
from parameters import *

import math

class GameUi():
    def __init__(self, board):
        self.screen = pygame.display.set_mode((900, 700))
        rect = pygame.draw.rect(self.screen, BG_COLOR, (0, 0, 900, 700))
        pygame.display.update(rect)

        count = len(board)
        # width of each block is 700/count
        self.width = math.floor(700/count)

    # move this to a 'draw panel' sub function?
    def draw_score(self, score):
        myfont = pygame.font.SysFont("freesansbold.ttf", 32)
        
        text_surface = myfont.render("Score: {}".format(score), True, (0,0,0))
        self.screen.blit(text_surface, (705,10))
        pygame.display.flip()
    def draw_board(self, board):
        rects = []

        rect = pygame.draw.rect(self.screen, BG_COLOR, (0, 0, 900, 700))
        pygame.display.update(rect)

        for x, col in enumerate(board):
            for y, value in enumerate(col):
                if value == Cell.EMPTY:
                    rects.append(pygame.draw.rect(self.screen, EMPTY_COLOR, (x*self.width, y * self.width, self.width, self.width )))
                if value == Cell.FOOD:
                    rects.append(pygame.draw.rect(self.screen, FOOD_COLOR, (x*self.width, y * self.width, self.width, self.width )))
                if value == Cell.BODY:
                    rects.append(pygame.draw.rect(self.screen, BODY_COLOR, (x*self.width, y * self.width, self.width,self.width) ))

        pygame.display.update(rects)

if __name__ == "__main__":
    # pygame stuff
    pygame.init()
    pygame.font.init()
    
    game_engine = Engine(ROW_COUNT, ROW_COUNT)
    game_ui = GameUi()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    game_engine.set_direction('Left')
                if keys[pygame.K_RIGHT]:
                    game_engine.set_direction('Right')
                if keys[pygame.K_UP]:
                    game_engine.set_direction('Up')
                if keys[pygame.K_DOWN]:
                    game_engine.set_direction('Down')
            
        done, _, _ =  game_engine.step()
        
        if done == True:
            break

        game_ui.draw_board(game_engine.board)
        game_ui.draw_score(game_engine.snake_len)
        pygame.time.delay(150)
