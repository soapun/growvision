import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base.base_net import BaseNet
from .cifar10_modified import PositionalEncoding

class Dense(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 16

        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)

        self.pe = PositionalEncoding(2, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = x.view(x.size(0), 8, -1)
        x = self.pe(x)
        x = x.view(x.size(0), -1)

        return x


class Dense_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 16

        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 32)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)

        return x
