from Try.Model.Utils import *
from torchvision import models

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
        self.shortcut1 = shortcut(n_channels, 32)
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

class ResUnet34(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(ResUnet34,self).__init__()

        self.base_model=models.resnet50(pretrained=True)
        self.base_layers=list(self.base_model.children())
        filters=[64,256,512,1024,2048]

        self.firstconv=nn.Conv2d(n_channels,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.firstbn=self.base_model.bn1
        self.firstrelu=self.base_model.relu
        self.firstmaxpool=self.base_model.maxpool
        self.encoder1 = self.base_model.layer1
        self.encoder2 = self.base_model.layer2
        self.encoder3 = self.base_model.layer3
        self.encoder4 = self.base_model.layer4

        self.center = DecoderBlock(in_channels=filters[4], n_filters=filters[4], kernel_size=3, is_deconv=True)
        self.decoder4 = DecoderBlock(in_channels=filters[4]+filters[3], n_filters=filters[3], kernel_size=3, is_deconv=True)
        self.decoder3 = DecoderBlock(in_channels=filters[3]+filters[2], n_filters=filters[2], kernel_size=3, is_deconv=True)
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], n_filters=filters[1], kernel_size=3, is_deconv=True)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[1], n_filters=filters[1], kernel_size=3, is_deconv=True)

        self.finalconv=nn.Sequential(nn.Conv2d(filters[1],32,3,padding=1,bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1,False),
                                     nn.Conv2d(32,n_classes,1))
        pass

    def forward(self,x):
        x=self.firstconv(x)
        x=self.firstbn(x)
        x=self.firstrelu(x)
        x_=self.firstmaxpool(x)

        e1=self.encoder1(x_)
        e2=self.encoder2(e1)
        e3=self.encoder3(e2)
        e4=self.encoder4(e3)

        center=self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f=self.finalconv(d1)
        f=torch.sigmoid(f)
        return f










