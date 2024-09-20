import numpy as np
import torch

import deep_q_learning
from engine import Engine
from parameters import *
from game_ui import GameUi
import pygame

neural_net = deep_q_learning.simple_net()
neural_net.load_state_dict(torch.load("model/net"))    

@torch.no_grad()
def get_action(state): # get the action (and apply epsilon)

    # convert state to a tensor
    state_tensor = torch.tensor(np.array([state], dtype=np.float32))
    
    q_values = neural_net(state_tensor)

    return np.argmax(q_values)

# run a few games with the final result
for i in range(10):
    # create game engine
    pygame.init()
    pygame.font.init()

    game_engine = Engine(ROW_COUNT, ROW_COUNT)
    game_ui = GameUi()

    cur_state = game_engine.get_state()
    done = False

    while not done :
        # map agent action to direction
        action = get_action(game_engine.get_state())
        if action == 0:
                game_engine.set_direction('Down')
        if action == 1:
                game_engine.set_direction('Up')
        if action == 2:
                game_engine.set_direction('Left')
        if action == 3:
                game_engine.set_direction('Right')

        done, reward, new_state = game_engine.step()

        pygame.time.delay(150)
        game_ui.draw_board(game_engine.board)
        game_ui.draw_score(game_engine.snake_len)