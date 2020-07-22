import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function



class Paraphraser(nn.Module):
    def __init__(self,in_planes, planes, stride=1):
        super(Paraphraser, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
  #      self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes,in_planes , kernel_size=3, stride=1, padding=1, bias=True)
  #      self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
  #      self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
  #      self.bn0_de = nn.BatchNorm2d(planes)
        self.deconv0 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
  #      self.bn1_de = nn.BatchNorm2d(in_planes)
        self.deconv1 = nn.ConvTranspose2d(planes,in_planes, kernel_size=3, stride=1, padding=1, bias=True)
  #      self.bn2_de = nn.BatchNorm2d(in_planes)
        self.deconv2 = nn.ConvTranspose2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)




#### Mode 0 - throw encoder and decoder (reconstruction)
#### Mode 1 - extracting teacher factors
    def forward(self, x,mode):

        if mode == 0:
        ## encoder
         out = self.leakyrelu((self.conv0(x)))
         out = self.leakyrelu((self.conv1(out)))
         out = self.leakyrelu((self.conv2(out)))
        ## decoder
         out = self.leakyrelu((self.deconv0(out)))
         out = self.leakyrelu((self.deconv1(out)))
         out = self.leakyrelu((self.deconv2(out)))


        if mode == 1:
         out = self.leakyrelu((self.conv0(x)))
         out = self.leakyrelu((self.conv1(out)))
         out = self.leakyrelu((self.conv2(out)))

        ## only throw decoder
        if mode == 2:
         out = self.leakyrelu((self.deconv0(x)))
         out = self.leakyrelu((self.deconv1(out)))
         out = self.leakyrelu((self.deconv2(out)))
        return out


class Translator(nn.Module):
    def __init__(self,in_planes, planes, stride=1):
        super(Translator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.1)
  #      self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = nn.Conv2d(in_planes,in_planes , kernel_size=3, stride=1, padding=1, bias=True)
  #     self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
  #     self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.leakyrelu((self.conv0(x)))
        out = self.leakyrelu((self.conv1(out)))
        out = self.leakyrelu((self.conv2(out)))
        return out




