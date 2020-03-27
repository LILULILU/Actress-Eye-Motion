import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random

class DQN(nn.Module):

    def __init__(self, num_hidden, num_action, num_input):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.bn = nn.BatchNorm1d(num_hidden)
        self.layer2 = nn.Linear(num_hidden, num_hidden)
        self.bn2 = nn.BatchNorm1d(num_hidden)
        self.head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.bn(x)
        x = F.relu(self.layer2(x))
        x = self.bn2(x)
        x = self.head(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

