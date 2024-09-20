import collections
import numpy as np
import torch

ReplayEvent = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    def __init__(self, capacity = 10000) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, event):
        self.buffer.append(event)
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)