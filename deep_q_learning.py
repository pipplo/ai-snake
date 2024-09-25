import torch

import torch.nn as nn

class simple_net(nn.Module):
    def __init__(self):
        super(simple_net, self).__init__()

        # Input is the data from the state at Engine::get_state
        # 8 input values
        self.model = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=.005)

        # TODO Initialize weights?

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
