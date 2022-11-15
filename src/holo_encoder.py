import torch
import torch.nn as nn
import math

class Down(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),

            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),

            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        )

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.skip(x)
        out3 = out1+out2
        out4 = self.net2(out3)
        out5 = out3+out4
        return out5

class Up(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,output_padding=1),

            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.net2 = nn.Sequential(
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0)
        )

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.skip(x)
        out3 = out1 + out2
        out4 = self.net2(out3)
        out5 = out3 + out4
        return out5

class HoloEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.netdown1 = Down(2,16)
        self.netdown2 = Down(16,32)
        self.netdown3 = Down(32,64)
        self.netdown4 = Down(64,96)
        self.netup0 = Up(96,64)
        self.netup1 = Up(64,32)
        self.netup2 = Up(32,16)
        self.netup3 = Up(16,1)
        self.norm = torch.nn.BatchNorm2d(1)
        self.tanH = torch.nn.Hardtanh(-math.pi, math.pi)

    def forward(self, x):

        out1=self.netdown1(x)
        out2=self.netdown2(out1)
        out3=self.netdown3(out2)
        out4=self.netdown4(out3)

        out5=self.netup0(out4)
        outa = out5 + out3
        out6 = self.netup1(outa)
        outb = out6 + out2
        out7 = self.netup2(outb)
        outc = out7 + out1
        out8 = self.netup3(outc)
        # hologram output
        out8 = self.norm(out8)
        out8 = self.tanH(out8)

        return out8

