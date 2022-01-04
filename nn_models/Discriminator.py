import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class Discriminator(nn.Module):
    def __init__(self, ngpu ,input_channels, input_size):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channels, input_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size) x 32 x 32
            nn.Conv2d(input_size, input_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size*2) x 16 x 16
            nn.Conv2d(input_size * 2, input_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size*4) x 8 x 8
            nn.Conv2d(input_size * 4, input_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (input_size*8) x 4 x 4
            nn.Conv2d(input_size * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
