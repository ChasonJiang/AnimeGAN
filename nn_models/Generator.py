import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, ngpu,input_channels,output_channels,output_size,):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_channels, output_size * 8, kernel_size = 4, stride = 1, padding= 0, bias=False),
            nn.BatchNorm2d(output_size * 8),
            nn.ReLU(True),
            # state size. (output_size*8) x 4 x 4
            nn.ConvTranspose2d(output_size * 8, output_size * 4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(output_size * 4),
            nn.ReLU(True),
            # state size. (output_size*4) x 8 x 8
            nn.ConvTranspose2d( output_size * 4, output_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size * 2),
            nn.ReLU(True),
            # state size. (output_size*2) x 16 x 16
            nn.ConvTranspose2d( output_size * 2, output_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_size),
            nn.ReLU(True),
            # state size. (output_size) x 32 x 32
            nn.ConvTranspose2d( output_size, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# class Generator(nn.Module):
#     def __init__(self,input_channels=100):
        
#         self.input_layer = nn.ConvTranspose2d(input_channels, 1024, kernel_size=4, stride=0)
#         self.block_layer = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
#         self.out_layer =  nn.Sequential(
#             nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2),
#             nn.BatchNorm2d(3),
#             nn.tanh()
#         )
    
#     def forward(self,x):
#         input_map = self.input_layer(x)
#         block_map = self.block_layer(input_map)
#         output = self.output_layer(block_map)
#         return output