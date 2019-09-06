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
    mCSV=mCSV.iloc[:,:].values
    return mCSV

def split_to_train_val(mCSV,rate,shuffle=False):
    """
    Read in csv and split data into train and val
    :param path:
    :param rate:
    :return:
    """
    imgs=read_in_csv(mCSV)
    if(shuffle):
        pass
    train=[]
    val=[]

    if rate==0:
        train=imgs
        print("Dataset Initialization finished: Train {0}, Val 0".format(train.shape[0]))
        val=np.array(val)
    else:
        x,y=imgs.shape
        num=int(x*rate)
        if num==0:
            train=imgs
            val=np.array(val)
        else:
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

class Dataset_WithLiver(Dataset):
    def __init__(self,data,classes,is_train,randomize):
        self.imgs=data
        self.classes=classes
        self.is_train=is_train
        self.randomize=randomize
        pass

    def __getitem__(self, index):

        imgPath,maskPath=self.imgs[index]
        img=cv2.imread(imgPath)[:,:,0]
        img=img/255

        # input=np.zeros((2,img.shape[0],img.shape[1]))
        # input[0, :, :] = img


        mask=cv2.imread(maskPath)[:,:,0]

        if self.randomize and self.is_train:
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

        mask[mask==1]=0
        mask[mask==2]=1

        input = img[np.newaxis, :, :]

        mask = LabelToOnehot(mask,self.classes)
        input=np.ascontiguousarray(input,dtype='float64')
        sample={'img':torch.from_numpy(input),'mask':torch.from_numpy(mask)}
        return sample


    def __len__(self):
        return self.imgs.shape[0]