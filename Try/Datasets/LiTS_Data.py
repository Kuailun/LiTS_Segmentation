from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import cv2
import numpy as np

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


class mDataset(Dataset):
    def __init__(self,data,classes):
        self.imgs=data
        self.classes=classes
        pass

    def __getitem__(self, index):

        imgPath,maskPath=self.imgs[index]
        img=cv2.imread(imgPath)[:,:,0]
        img=img/255

        img = img[np.newaxis,:,:]
        mask=cv2.imread(maskPath)[:,:,0]

        mask = LabelToOnehot(mask,self.classes)
        sample={'img':torch.from_numpy(img),'mask':torch.from_numpy(mask)}
        return sample


    def __len__(self):
        return self.imgs.shape[0]