from Try.Model.mModel import *
import os
import mPath
from Datasets import LiTS_Data
import cv2
import numpy as np
from Try import Loss, Utils as ut
import torch
import Preprocess as pp

GPU_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_DEVICES
Use_GPU=torch.cuda.is_available()
Output_Class=2
Train_Epochs=10
Train_Batch_Size=10
Validation_Percent=0.1
Save_CheckPoint=True
Output_per_epoch=1
learning_rate=0.01
weights=[0,1,0]
predict_mode=1  #1-预测一个patch, #2-预测一个layer, #3-预测一个nii

def predict_patch(net,img,gpu):
    net.eval()
    img=img[np.newaxis,np.newaxis,:,:]
    img=torch.from_numpy(img)

    img = img.float()
    if gpu:
        img=img.cuda()
        pass

    img = img.float()

    # 只有训练阶段才追踪历史
    with torch.set_grad_enabled(False):
        output = net(img)
        _, preds = torch.max(output, 1)
        ut.plot_img(preds[0, :, :], mPath.DataPath_Volume_Predict + "output.jpg", 'Output', 2)
        pass
    pass

def predict_layer(net,layer,gpu):
    net.eval()
    samples = pp.make_patch(layer, resize=(128, 128), patch=5)

    patches=samples.shape[2]
    ret_samples=np.zeros_like(samples)
    for patch in range(patches):
        img=samples[:,:,patch]
        img = img[np.newaxis, np.newaxis, :, :]
        img = torch.from_numpy(img)

        img = img.float()
        if gpu:
            img = img.cuda(0)
            pass

        img = img.float()

        # 只有训练阶段才追踪历史
        with torch.set_grad_enabled(False):
            output = net(img)
            _, preds = torch.max(output, 1)
            ret_samples[:,:,patch]=preds[0,:,:].cpu().numpy()
            ut.plot_img(preds[0, :, :], mPath.DataPath_Volume_Predict + "output"+str(patch)+".jpg", 'Output', 2)
            pass
        pass

    preds=ut.Merge_Patches(ret_samples,[512,512],5)
    ut.save_img(preds,mPath.DataPath_Volume_Predict + "output.jpg")







# predict_Nii()
# predict_Layer()
# predict_Patch()

if __name__=='__main__':
    net=UNet_Yading(n_channels=1,n_classes=Output_Class)
    # dummy_input=torch.rand(Train_Batch_Size,1,256,256)
    # writer.add_graph(net,input_to_model=(dummy_input,))

    if Use_GPU:
        net.cuda()
        net.load_state_dict(torch.load(mPath.DataPath_Net_Predict))
        print('Using GPU')
    else:
        net.cpu()
        net.load_state_dict(torch.load(mPath.DataPath_Net_Predict,map_location='cpu'))
        print('Using CPU')

    print("Pretrained model loaded")

    mCSV=LiTS_Data.read_in_csv(mPath.CSVPath+"predict.csv")

    if predict_mode==1:
        img=cv2.imread('F:/Workspace/python/Data/Data_LiTS/volume/volume-0/1-2.jpg')[:,:,0]/255
        mask=predict_patch(net,img,Use_GPU)
        pass
    if predict_mode==2:
        img=cv2.imread('F:/Workspace/python/Data/Data_LiTS/volume/volume-0/57.jpg')[:,:,0]/255
        samples=predict_layer(net,img,Use_GPU)


