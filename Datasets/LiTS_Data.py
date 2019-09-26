from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import cv2
import numpy as np
import random
import torchvision

def read_in_csv(path):
    if not(os.path.exists(path)):
        raise(path+" is not existed")
    mCSV=pd.read_csv(path)
    mCSV=mCSV.iloc[:,:].values
    return mCSV

def split_to_train_val(mCSV,mode,rate,shuffle=False):
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
            if mode=='Multi':
                if len(imgs)>5000:
                    train=imgs[0:-4982]
                    val=imgs[-4982:-1]
                else:
                    train=imgs
                    val=np.array(val)
            elif mode=='Single':
                num=16930
                val=imgs[num:-1]
                train=imgs[0:num]
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
    def __init__(self,data,train_mode,classes,is_train,randomize):
        self.imgs=data
        self.classes=classes
        self.is_train=is_train
        self.randomize=randomize
        self.train_mode=train_mode
        pass

    def __getitem__(self, index):

        if self.train_mode=='Single':
            imgPath,maskPath=self.imgs[index]
            img=cv2.imread(imgPath)[:,:,0]
            img=np.array(img,dtype='float64')
            img=img/255

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
            ma=np.copy(mask)
            ma[mask==2]=1
            ma=np.array(ma,dtype='float64')
            img=img*ma

            mask[mask==1]=0
            mask[mask==2]=1

            input = img[np.newaxis, :, :]

            mask = LabelToOnehot(mask,self.classes)
            input=np.ascontiguousarray(input,dtype='float64')
            sample={'img':torch.from_numpy(input),'mask':torch.from_numpy(mask)}
            return sample
        elif self.train_mode=='Multi':
            imgPaths= self.imgs[index]
            imgPath=imgPaths[0:-1]
            maskPath=imgPaths[-1]

            mask=cv2.imread(maskPath)[:,:,0]
            width,height=mask.shape
            input=np.zeros((len(imgPath),width,height),dtype='uint8')
            for i in range(len(imgPath)):
                input[i,:,:]=cv2.imread(imgPath[i])[:,:,0]
                pass

            input=np.array(input,dtype='float64')
            input=input/255

            if self.randomize and self.is_train:
                if (GetRandom(0.1)):
                    # 水平镜像
                    # cv2.imshow('1', input[0,:, :])
                    # cv2.waitKey(0)
                    # cv2.imshow('2', mask * 127)
                    # cv2.waitKey(0)
                    for i in range(input.shape[0]):
                        input[i,:,:]=cv2.flip(input[i,:,:],1)
                    mask = cv2.flip(mask, 1)
                    # cv2.imshow('1', input[0,:, :])
                    # cv2.waitKey(0)
                    # cv2.imshow('2', mask * 127)
                    # cv2.waitKey(0)
                    pass

                if (GetRandom(0.1)):
                    # 垂直镜像
                    # cv2.imshow('1', input[0,:, :])
                    # cv2.waitKey(0)
                    # cv2.imshow('2', mask * 127)
                    # cv2.waitKey(0)
                    for i in range(input.shape[0]):
                        input[i, :, :] = cv2.flip(input[i, :, :], 0)
                    mask = cv2.flip(mask, 0)
                    # cv2.imshow('1', input[0,:, :])
                    # cv2.waitKey(0)
                    # cv2.imshow('2', mask * 127)
                    # cv2.waitKey(0)
                    pass

                if (GetRandom(0.2)):
                    # 旋转90度
                    # cv2.imshow('1', input[0,:, :])
                    # cv2.waitKey(0)
                    # cv2.imshow('2', mask * 127)
                    # cv2.waitKey(0)
                    for i in range(input.shape[0]):
                        input[i, :, :] = np.rot90(input[i,:,:], 1)
                    mask = np.rot90(mask, 1)
                    # cv2.imshow('1',input[0,:, :])
                    # cv2.waitKey(0)
                    # cv2.imshow('2',mask*127)
                    # cv2.waitKey(0)
                    pass
                pass

            # cv2.imshow('1', input[0,:, :])
            # cv2.waitKey(0)

            mask[mask == 1] = 0
            mask[mask == 2] = 1

            mask = LabelToOnehot(mask, self.classes)
            input = np.ascontiguousarray(input, dtype='float64')
            sample = {'img': torch.from_numpy(input), 'mask': torch.from_numpy(mask)}
            return sample


    def __len__(self):
        return self.imgs.shape[0]

    pass


class Dataset_Liver(Dataset):
    def __init__(self, data, is_train, randomize):
        self.imgs = data
        self.is_train = is_train
        self.randomize = randomize
        pass

    def __getitem__(self, index):

        imgPath, maskPath = self.imgs[index]
        img = cv2.imread(imgPath)[:, :, 0]
        img = img / 255

        mask = cv2.imread(maskPath)[:, :, 0]

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

            # if (GetRandom(0.1)):
            #     # 调整图像亮度
            #     rate=random.randint(87,107)
            #     img=img*rate/100
            #     img[img>1.0]=1.0
            pass

        mask[mask == 2] = 1

        # cv2.imshow('1', img)
        # cv2.waitKey(0)

        input=cv2.resize(img,(256,256))
        mask=cv2.resize(mask,(256,256))

        # inpu=np.zeros((3,256,256),dtype='float64')
        # inpu[0,:,:]=input
        # inpu[1,:,:]=input
        # inpu[2,:,:]=input
        input = input[np.newaxis, :, :]

        # input=inpu


        mask = LabelToOnehot(mask, 2)
        input = np.ascontiguousarray(input, dtype='float64')
        sample = {'img': torch.from_numpy(input) , 'mask': torch.from_numpy(mask)}
        return sample

    def __len__(self):
        return self.imgs.shape[0]

    pass