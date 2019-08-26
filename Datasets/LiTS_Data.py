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
    else:
        x,y=imgs.shape
        num=int(x*rate)
        train=imgs[num:-1]
        val=imgs[0:num]
        pass
    print("Dataset Initialization finished: Train {0}, Val {1}".format(train.shape[0], val.shape[0]))
    return train,val

def LabelToOnehot(img,classes):
    """
    Convert label to onehot label
    :param img:
    :param classes:
    :return:
    """
    if classes==1:
        tu=(0)
    elif classes==2:
        tu=(0,1)
    elif classes==3:
        tu=(0,1,2)
    onehot=np.zeros((classes,img.shape[0],img.shape[1]))
    for i,l in enumerate(tu):
        onehot[i,:,:]=img==l
        pass
    return onehot

class mDataset(Dataset):
    def __init__(self,data,classes):
        self.imgs=data
        self.classes=classes
        pass

    def __getitem__(self, index):
        # if self.shuffle:
        #     np.random.shuffle(self.imgs)
        #     pass
        imgPath,maskPath=self.imgs[index]
        img=cv2.imread(imgPath)[:,:,0]
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        img = img[np.newaxis,:,:]
        mask=cv2.imread(maskPath)[:,:,0]

        # cv2.imshow('mask',mask*127)
        # cv2.waitKey()
        mask = LabelToOnehot(mask,self.classes)
        sample={'img':torch.from_numpy(img),'mask':torch.from_numpy(mask)}
        return sample


    def __len__(self):
        return self.imgs.shape[0]