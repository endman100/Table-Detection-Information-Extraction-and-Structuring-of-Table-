import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 4*8)
        self.down1 = Down(4*8, 4*16)
        self.down2 = Down(4*16, 4*32)
        self.down3 = Down(4*32, 4*64)
        self.down4 = Down(4*64, 4*64)
        self.up1 = Up(4*128, 4*32, bilinear)
        self.up2 = Up(4*64, 4*16, bilinear)
        self.up3 = Up(4*32, 4*8, bilinear)
        self.up4 = Up(4*16, 4*8, bilinear)
        self.outc = OutConv(4*8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # x = self.admp(x5).view(-1)
        # x =  F.relu(self.full1(x))
        # x = self.full2(x)
        return logits#, x
if __name__ == '__main__':
    import sys
    sys.path.append("..")
    model = UNet(3, 1).cuda()
    lossFunctionBCE = torch.nn.BCELoss()
    lossFunctionMSE = torch.nn.MSELoss() 
    for i in range(300):    
        print(i)
        img = torch.rand((1,3,1000,1000)).cuda()
        target = torch.rand((1,1,1000,1000)).cuda()
        angleAnswer = torch.rand((1)).cuda()
        output, angle = model(img)
        print(output.shape)
        print(target.shape)
        print(angle.shape)
        output = output.view(-1).unsqueeze(1).unsqueeze(1)
        target = target.view(-1).unsqueeze(1).unsqueeze(1).float()
        loss = lossFunctionMSE(angle, angleAnswer)/2 + lossFunctionBCE(output, target)/2
        loss.backward()
