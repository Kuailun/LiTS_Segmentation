from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import cv2
import numpy as np
import random

def read_in_csv(path):
    if not(os.path.exists(path)):
        raise(path+" is not existed")
    mCSV=pd.read_csv(path)
    return mCSV

def split_to_train_val(mCSV,rate,shuffle=False):
    """
    Read in csv and split data into train and val
    :param path:
    :param rate:
    :return:
    """
    imgs=read_in_csv(mCSV)
    imgs=imgs.iloc[:,:].values
    if(shuffle):
        pass
    train=[]
    val=[]

    if rate==0:
        train=imgs
        print("Dataset Initialization finished: Train {0}, Val 0".format(train.shape[0]))
        val=[]
    else:
        x,y=imgs.shape
        num=int(x*rate)
        train=imgs[num:-1]
        val=imgs[0:num]
        print("Dataset Initialization finished: Train {0}, Val {1}".format(train.shape[0], val.shape[0]))
        pass

    return train,val

def LabelToOnehot(img,classes):
    """
    Convert label to onehot label
    :param img:
    :param classes:
    :return:
    """
    if classes==2:
        img[img==2]=1
        pass

    onehot = np.zeros((classes, img.shape[0], img.shape[1]))
    for i in range(classes):
        onehot[i,:,:]=img==i

    return onehot

def GetRandom(rate):
    s=random.randint(1,100)
    if s<rate*100:
        return True
    else:
        return False

class mDataset(Dataset):
    def __init__(self,data,classes,randomize):
        self.imgs=data
        self.classes=classes
        self.randomize=randomize
        pass

    def __getitem__(self, index):

        imgPath,maskPath=self.imgs[index]
        img=cv2.imread(imgPath)[:,:,0]
        img=img/255

        img = img[np.newaxis,:,:]
        mask=cv2.imread(maskPath)[:,:,0]

        if self.randomize:
            if (GetRandom(0.1)):
                # 水平镜像
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
                pass

            if (GetRandom(0.1)):
                # 垂直镜像
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
                pass

            if (GetRandom(0.2)):
                # 旋转90度
                img = np.rot90(img, 1)
                mask = np.rot90(mask, 1)
                pass
            pass

        mask = LabelToOnehot(mask,self.classes)
        sample={'img':torch.from_numpy(img),'mask':torch.from_numpy(mask)}
        return sample


    def __len__(self):
        return self.imgs.shape[0]