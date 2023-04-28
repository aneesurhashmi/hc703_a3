import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels), # good practice to use batch norm
            nn.ReLU(inplace=True), # inplace=True means it will modify the input directly
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):

    def __init__(self, n_channels = 3, n_classes = 2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.Sequential(nn.ConvTranspose2d(1024, 512+256, kernel_size=2, stride=2), DoubleConv(512+256, 512))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512+512, 512, kernel_size=2, stride=2), DoubleConv(512, 256))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256+256, 256 , kernel_size=2, stride=2), DoubleConv(256, 128))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128+128, 128, kernel_size=2, stride=2), DoubleConv(128, 64))
        self.outc = nn.Sequential(nn.Conv2d(64+64, 64, kernel_size=1), nn.Conv2d(64, n_classes, kernel_size=1))

    def forward(self, x):
        # Encoding path
        x1 = self.inc(x) # (64, x.shape[0], x.shape[1]) : (64, 256, 256)
        x2 = self.down1(x1) # (128, x.shape[0], x.shape[1]/2): (128, 256, 128)
        x3 = self.down2(x2) # (256, x.shape[0]/4, x.shape[1]/4): (256, 128, 64)
        x4 = self.down3(x3) # (512, x.shape[0]/8, x.shape[1]/8): (512, 64, 64)
        x5 = self.down4(x4) # (1024, x.shape[0]/16, x.shape[1]/16): (1024, 16, 16)

        # Decoding path
        x = self.up1(x5) # (512, x.shape[0]/8, x.shape[1]/8): (512, 32, 32)
        x = torch.cat([x4, x], dim=1)
        x = self.up2(x) # # (256, x.shape[0]/4, x.shape[1]/4): (256, 64, 64)
        x = torch.cat([x3, x], dim=1)
        x = self.up3(x) # (128, x.shape[0]/2, x.shape[1]/2): (128, 128, 128)
        x = torch.cat([x2, x], dim=1)
        x = self.up4(x) # (64, x.shape[0], x.shape[1]): (64, 256, 256)
        x = torch.cat([x1, x], dim=1)

        # x = F.sigmoid(self.outc(x)) # (2, x.shape[0], x.shape[1]): (2, 256, 256)
        x = self.outc(x) # (2, x.shape[0], x.shape[1]): (2, 256, 256)
        return x


