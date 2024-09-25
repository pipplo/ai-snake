import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import pickle

from engine import Engine
from parameters import *
from game_ui import GameUi
import pygame

class SimpleAgent:
    def __init__(
            self,
            learning_rate=.005,
            start_epsilon=1,
            epsilon_decay=.1,
            final_epsilon=0,
            discount_factor = .9,
            ):
        
        # create a dict representing the q table. It's a 'default dict' that will default to the
        # given input entry
        self.q_values = defaultdict(lambda: np.zeros(4)) # TODO: Paramaterize the action count. 

        # Will paramaterize some of these later
        self.lr = learning_rate
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor # sometimes referred to as gamma

    def get_action(self, state): # get the action (and apply epsilon)

        # first let's see if we should do a random selection
        if np.random.random() < self.epsilon:
            return random.randrange(4)
        
        q_values = self.q_values[state]

        return np.argmax(q_values)
        

    def update(
            self,
            cur_state,
            action,
            reward,
            terminated,
            new_state,
            ): # update the q table based on results of last action

        # This uses a bunch of math from different locations
        future_q = (not terminated) * np.max(self.q_values[new_state]) 

        temporal_difference = ( reward + self.discount_factor * future_q - self.q_values[cur_state][action] )

        self.q_values[cur_state][action] = self.q_values[cur_state][action] + (self.lr * temporal_difference)

        #print(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def load(self, file):
        with open(file, 'rb') as f:
            simplified = pickle.load(f)
            self.q_values.clear()
            for k,v in simplified.items():
                self.q_values[k] = v


    def save(self, file):
        simplified = {}
        for k,v in self.q_values.items():
            simplified[k] = v

        with open(file, 'wb') as f:
            pickle.dump(simplified, f)

def train():
    n_episodes = 100000

    learning_rate = 0.005
    start_epsilon = 1.0
    final_epsilon = .01
    epsilon_decay = start_epsilon / (n_episodes/2)

    agent = SimpleAgent(learning_rate=learning_rate, start_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

    max_score = 0
    max_step_count = 0

    for episode in tqdm(range(n_episodes)):
    #for episode in range(n_episodes):

        # create game engine
        game_engine = Engine(ROW_COUNT, ROW_COUNT)

        cur_state = game_engine.get_state()
        done = False

        while not done :
            # map agent action to direction
            action = agent.get_action(game_engine.get_state())
            if action == 0:
                 game_engine.set_direction('Down')
            if action == 1:
                 game_engine.set_direction('Up')
            if action == 2:
                 game_engine.set_direction('Left')
            if action == 3:
                 game_engine.set_direction('Right')

            done, reward, new_state = game_engine.step()
            agent.update(cur_state, action, reward, done, new_state)
            cur_state = new_state


        agent.decay_epsilon()

        max_score = max(game_engine.snake_len, max_score)
        max_step_count = max(game_engine.step_count, max_step_count)
            
    print((max_score, max_step_count))

    return agent

def play(agent):
    # run a few games with the final result
    for i in range(10):
        # create game engine
        pygame.init()
        pygame.font.init()

        game_engine = Engine(ROW_COUNT, ROW_COUNT)
        game_ui = GameUi(game_engine.board)

        cur_state = game_engine.get_state()
        done = False

        agent.epsilon = 0
        while not done :
            # map agent action to direction
            action = agent.get_action(game_engine.get_state())
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

# a default way to run the agent to train itself
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a simple Q-Table agent")
    parser.add_argument('--play', dest='play', action='store_false', help='Play the agent')
    parser.add_argument('--train', dest='train', action='store_true', help='train the model')
    parser.add_argument('--load', dest='load_file', action='store', default='qtable/qtable.pkl', help='specified file to load', )
    args = parser.parse_args()

    if (args.train):
        agent = train()
        agent.save(args.load_file)
    
    agent = SimpleAgent()

    agent.load(args.load_file)

    if(args.play):
        play(agent)
