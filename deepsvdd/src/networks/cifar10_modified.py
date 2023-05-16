import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base.base_net import BaseNet

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x

class CIFAR10_LeNet_3D(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2))

        self.conv1 = nn.Conv3d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm3d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv3d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm3d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv3d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm3d(128, eps=1e-04, affine=False)
        self.pe = PositionalEncoding(128 * 4 * 4, 8)

        self.fc1 = nn.Linear(128 * 4 * 4 * 8, self.rep_dim, bias=False)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), x.size(2), -1)
        x = self.pe(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        return x


class CIFAR10_LeNet_Autoencoder_3D(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv3d(3, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm3d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv3d(32, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm3d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv3d(64, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm3d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4 * 8, self.rep_dim, bias=False)
        #self.pe = PositionalEncoding(128 * 4 * 4, 8)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.Sequential(
            nn.Conv3d(int(self.rep_dim / (4 * 4 * 8)), 128, 5, bias=False, padding=2),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )

        self.deconv2 = nn.Sequential(
            nn.Conv3d(128, 64, 5, bias=False, padding=2),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )

        self.deconv3 = nn.Sequential(
            nn.Conv3d(64, 32, 5, bias=False, padding=2),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )

        self.deconv4 = nn.Sequential(
            nn.Conv3d(32, 3, 5, bias=False, padding=2),
            #nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )


        #self.deconv1 = nn.ConvTranspose3d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm3d(128, eps=1e-04, affine=False)
        #self.deconv2 = nn.ConvTranspose3d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm3d(64, eps=1e-04, affine=False)
        #self.deconv3 = nn.ConvTranspose3d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm3d(32, eps=1e-04, affine=False)
        #self.deconv4 = nn.ConvTranspose3d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4[0].weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))

        # x = x.view(x.size(0), x.size(2), -1)
        # x = self.pe(x)
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4 * 8)), 8, 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.leaky_relu(self.bn2d4(x))
        x = self.deconv2(x)
        x = F.leaky_relu(self.bn2d5(x))
        x = self.deconv3(x)
        x = F.leaky_relu(self.bn2d6(x))
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x
