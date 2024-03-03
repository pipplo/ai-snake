import pygame

from engine import Engine
from parameters import *

class GameUi():
    def __init__(self):
        self.screen = pygame.display.set_mode((800, 600))
        rect = pygame.draw.rect(self.screen, 0x000000, (0, 0, 800, 600))
        pygame.display.update(rect)

    # move this to a 'draw panel' sub function?
    def draw_score(self, score):
        myfont = pygame.font.SysFont("freesansbold.ttf", 32)
        
        text_surface = myfont.render("Score: {}".format(score), True, 0xffffffff, (0,0,0))
        self.screen.blit(text_surface, (605,10))
    def draw_board(self, board):
        rects = []

        for x, col in enumerate(game_engine.board):
            for y, value in enumerate(col):
                if value == 0:
                    rects.append(pygame.draw.rect(self.screen, 0x1B1212, (x*50+5, y * 50+5, 40, 40 )))
                if value == 1:
                    rects.append(pygame.draw.rect(self.screen, 0x151b54, (x*50+5, y * 50+5, 40, 40 )))
                if value == 2:
                    rects.append(pygame.draw.rect(self.screen, 0x002200, (x*50 + 5, y * 50 + 5, 40,40) ))

                # if board[x][y] == 0:
                #     rects.append(pygame.draw.rect(screen, 0x1B1212, (x*50+5, y * 50+5, 40, 40 )))
                # if board[x][y] == 1:
                #     rects.append(pygame.draw.rect(screen, 0x151b54, (x*50+5, y * 50+5, 40, 40 )))
                # rects.append(pygame.draw.rect(screen, 0x151b54, (x*50, y * 50, 50, 50) ))
                # rects.append(pygame.draw.rect(screen, 0x0000FF, (x*50+5, y * 50+5, 40, 40 )))
                #rects.append(pygame.draw.rect(screen, 0x353935, (x*50, y * 50, 50, 50) ))
                    
        pygame.display.update(rects)

if __name__ == "__main__":
    # pygame stuffs
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
            
        if game_engine.step() == False:
            break

        game_ui.draw_board(game_engine.board)
        game_ui.draw_score(game_engine.snake_len)
        pygame.time.delay(150)
