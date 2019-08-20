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

class mDataset(Dataset):
    def __init__(self,mCSV,shuffle=False):
        self.imgs=read_in_csv(mCSV)
        self.imgs=self.imgs.iloc[:,:].values
        self.shuffle=shuffle
        pass
    pass

    def __getitem__(self, index):
        # if self.shuffle:
        #     np.random.shuffle(self.imgs)
        #     pass
        imgPath,maskPath=self.imgs[index]
        img=cv2.imread(imgPath)[:,:,0]
        img = img[np.newaxis,np.newaxis,:,:]
        mask=cv2.imread(maskPath)[:,:,0]
        mask = mask[np.newaxis,np.newaxis,:,:]
        sample={'img':torch.from_numpy(img),'mask':torch.from_numpy(mask)}
        return sample


    def __len__(self):
        return self.imgs.shape[0]