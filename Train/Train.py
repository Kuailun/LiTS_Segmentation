from Datasets import LiTS_Data
from Model.mModel import UNet
from torch import optim,nn
import mPath
import torch
import cv2
import pandas as pd

mDataset = LiTS_Data.mDataset(mPath.CSVPath + "data.csv")

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.01,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=0.5):
    optimizer=optim.SGD(net.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=0.0005)
    criterion=nn.BCELoss()

    # Begin training
    for epoch in range(epochs):
        print("Starting epoch{}/{}".format(epoch+1,epochs))
        net.train()

        epoch_loss=0
        for i in range(len(mDataset)):
            sample=mDataset[i]
            img=sample['img']
            mask=sample['mask']
            img=img.cuda()
            mask=mask.cuda()
            img=img.float()
            mask=mask.float()

            mask_pred=net(img)
            loss=criterion(mask_pred,mask)
            epoch_loss=epoch_loss+loss.item()
            print("epoch-{0},{1} loss={2}".format(epoch,i,loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    pass

net=UNet(n_channels=1,n_classes=1)
print(torch.cuda.is_available())
net.cuda()
train_net(net)

