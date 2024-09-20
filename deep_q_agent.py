import torch
import torch.nn as nn

import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import deep_q_learning
import deep_q_replay_buffer
from deep_q_replay_buffer import ReplayEvent

from engine import Engine
from parameters import *
from game_ui import GameUi
import pygame

class DeepQAgent:
    def __init__(
            self,
            model,
            replay_buffer,
            learning_rate,
            start_epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor = .9,
            ):
        
        self.model = model
        self.target_model = model
        self.replay_buffer = replay_buffer
        # Will paramaterize some of these later
        self.lr = learning_rate
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor # sometimes referred to as gamma

        self.train_count = 0

    @torch.no_grad()
    def get_action(self, state): # get the action (and apply epsilon)

        # first let's see if we should do a random selection
        if np.random.random() < self.epsilon:
            return random.randrange(4)
        
        # convert state to a tensor
        state_tensor = torch.tensor(np.array([state], dtype=np.float32))
        
        q_values = self.model(state_tensor)

        return np.argmax(q_values)
        

    def update(
            self,
            cur_state,
            action,
            reward,
            terminated,
            new_state,
            ):

        # The logic here is to update the replay buffer with every new transition observed
        # Then if we have a big enough replay buffer, then we can start to do some training
        # Then after enough training iterations we will periodically replace the target_network 
        # with the current network

        self.replay_buffer.append(ReplayEvent(cur_state, action, reward, terminated, new_state))

        if len(self.replay_buffer) < 10000: # don't start doing replay sampling until we have enough samples
            return
        
        self.train_step()

        if self.train_count % 10000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            torch.save(self.model.state_dict(), "model/net")
            print("Update")

    def calc_loss(self, batch, device="cpu"):
        # unpack batch
        states, actions, rewards, dones, next_states = batch

        # convert everything from batch to torch tensors and move it to device
        states_v = torch.tensor(states).to(device, dtype=torch.float32)
        next_states_v = torch.tensor(next_states).to(device, dtype=torch.float32)
        actions_v = torch.tensor(actions).to(device, dtype=torch.int64)
        rewards_v = torch.tensor(rewards).to(device, dtype=torch.float32)
        done_mask = torch.ByteTensor(dones).to(device)
        done_mask = done_mask.to(torch.bool)

        # get output from NNs which is used for calculating state action value with discount
        state_action_values = self.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_model(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.discount_factor + rewards_v
        # Calculate NN loss
        return nn.MSELoss()(state_action_values, expected_state_action_values)
    
    def train_step(self):
        self.model.optimizer.zero_grad()
        batch = self.replay_buffer.sample(5)
        loss_t = self.calc_loss(batch)
        loss_t.backward()
        self.model.optimizer.step()

        self.train_count += 1

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def train():
    n_episodes = 100000

    learning_rate = 0.005
    start_epsilon = 1.0
    final_epsilon = .01
    epsilon_decay = start_epsilon / (n_episodes/2)

    # TODO switch to loading the network
    neural_net = deep_q_learning.simple_net()
    replay_buffer = deep_q_replay_buffer.ReplayBuffer()

    torch.save(neural_net.state_dict(), "model/net")
    agent = DeepQAgent(model=neural_net, replay_buffer=replay_buffer, learning_rate=learning_rate, start_epsilon=start_epsilon, epsilon_decay=epsilon_decay, final_epsilon=final_epsilon)

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
        
    
    # print(agent.q_values)
    print((max_score, max_step_count))

    # run a few games with the final result
    for i in range(10):
        # create game engine
        pygame.init()
        pygame.font.init()

        game_engine = Engine(ROW_COUNT, ROW_COUNT)
        game_ui = GameUi()

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

    # print(agent.q_values)
    print((max_score, max_step_count))

# a default way to run the agent to train itself
if __name__ == '__main__':
    train()