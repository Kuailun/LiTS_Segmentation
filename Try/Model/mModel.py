from Try.Model.Utils import *

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        """
        Set parameters for model
        :param n_channels: 1-Grayscale, 3-RGB
        :param n_classes: label numbers
        """

        super(UNet,self).__init__()
        self.inc = inconv(n_channels, 64) #假设输入通道数n_channels为3，输出通道数为64
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

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
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x  # 进行二分类

    pass

class UNet_Yading(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        Set parameters for model
        :param n_channels: 1-Grayscale, 3-RGB
        :param n_classes: label numbers
        """

        super(UNet_Yading, self).__init__()
        self.inc = inconv(n_channels, 32)  # 假设输入通道数n_channels为3，输出通道数为64
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.up1 = up3(1024, 256)
        self.up2 = up3(512, 128)
        self.up3 = up3(256, 64)
        self.up4 = up3(128, 32)
        self.up5 = up3(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x  # 进行二分类
    pass