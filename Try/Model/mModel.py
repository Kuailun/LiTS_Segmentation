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

class ResUnet(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(ResUnet, self).__init__()
        self.shortcut1 = shortcut(1, 32)
        self.d_conv1=double_conv(n_channels,32)
        self.pool=nn.MaxPool2d(2)

        self.shortcut2=shortcut(32,64)
        self.d_conv2=double_conv(32,64)

        self.shortcut3=shortcut(64,128)
        self.d_conv3=tripple_conv(64,128)

        self.shortcut4=shortcut(128,256)
        self.d_conv4=tripple_conv(128,256)

        self.shortcut5=shortcut(256,512)
        self.d_conv5=tripple_conv(256,512)

        self.up4 = res_up3(1024, 256)
        self.up3 = res_up3(512, 128)
        self.up2 = res_up3(256, 64)
        self.up1 = res_up3(128, 32)

        self.outconv=nn.Conv2d(32,n_classes,1)

    def forward(self, x):
        x1=self.d_conv1(x)
        x1_res=self.shortcut1(x)+x1
        x1_res=self.pool(x1_res)

        x2=self.d_conv2(x1_res)
        x2_res=self.shortcut2(x1_res)+x2
        x2_res=self.pool(x2_res)

        x3=self.d_conv3(x2_res)
        x3_res=self.shortcut3(x2_res)+x3
        x3_res=self.pool(x3_res)

        x4=self.d_conv4(x3_res)
        x4_res=self.shortcut4(x3_res)+x4
        x4_res=self.pool(x4_res)

        x5=self.d_conv5(x4_res)
        x5_res=self.shortcut5(x4_res)+x5

        x = self.up4(x5_res, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x=self.outconv(x)
        x = torch.sigmoid(x)
        return x  # 进行二分类











