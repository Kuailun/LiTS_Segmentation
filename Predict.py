from Try.Model.mModel import *
import os,fnmatch
import mPath
from Datasets import LiTS_Data
import cv2
import numpy as np
from Try import Loss, Utils as ut
import torch
import Preprocess as pp
from medpy.io import load, save
import nibabel as nib

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
            # ut.plot_img(preds[0, :, :], mPath.DataPath_Volume_Predict + "output"+str(patch)+".jpg", 'Output', 2)
            pass
        pass

    preds=ut.Merge_Patches(ret_samples,[512,512],5)
    ut.save_img(preds,mPath.DataPath_Volume_Predict + "output.jpg")
    return preds

def predict_nii(net,nii,gpu,name):
    net.eval()
    nii=np.array(nii,dtype='float64')
    nii[nii < -200] = -200
    nii[nii > 250] = 250
    nii = ((nii + 200) * 255 // 450)
    nii = np.array(nii, dtype='uint8')

    layer=nii.shape[2]
    ut.CheckDirectory(mPath.DataPath_Volume_Predict+'temp/')
    for i in range(layer):
        save(nii[:,:,i], mPath.DataPath_Volume_Predict+'temp/'+str(i)+'.jpg')
        pass

    save_nii=np.zeros((512,512,layer),dtype='float32')
    for i in range(layer):
        current_img = cv2.imread(mPath.DataPath_Volume_Predict+'temp/'+str(i)+'.jpg')[:, :, 0]
        current_img=np.array(current_img,dtype='float64')
        current_img = current_img / 255
        samples = predict_layer(net, current_img, gpu)
        samples=np.flip(samples,0)
        samples = np.rot90(samples)
        samples = np.rot90(samples)
        samples = np.rot90(samples)
        save_nii[:,:,i]=samples
        print("Predicting {}-{}".format(name,i))
        pass

    new_img=nib.Nifti1Image(save_nii,affine=np.eye(4))
    nib.save(new_img,mPath.DataPath_Volume_Predict+name+'-.nii')

    #
    # layers = img1.shape[2]
    #
    # realstart = max(int(startposition * (1 - rate)) - slices // 2, 0)
    # realend = min(int((layers - endposition) * rate + endposition) + slices // 2, layers)
    #
    # for j in range(realend - realstart):
    #     index = j + realstart
    #
    #     im1 = np.array(img1[:, :, index])
    #     im2 = np.array(img2[:, :, index])
    #
    #     im1 = cv2.resize(im1, resize)
    #     im2 = cv2.resize(im2, resize)
    #
    #     save(im1, volume + 'volume-' + str(i) + '/' + str(index) + '.jpg')
    #     save(im2, mask + 'segmentation-' + str(i) + '/' + str(index) + '.jpg')
    #     print("Saving image " + str(i) + " " + str(index))
    #     pass
    # pass


pass

if __name__=='__main__':
    # net=UNet_Yading(n_channels=1,n_classes=Output_Class)
    net=torch.load(mPath.DataPath_Net_Final)
    # dummy_input=torch.rand(Train_Batch_Size,1,256,256)
    # writer.add_graph(net,input_to_model=(dummy_input,))

    if Use_GPU:
        net.cuda()
        # net.load_state_dict(torch.load(mPath.DataPath_Net_Predict))
        params=list(net.named_parameters())
        print('Using GPU')
    else:
        net.cpu()
        # net.load_state_dict(torch.load(mPath.DataPath_Net_Predict,map_location='cpu'))
        print('Using CPU')

    print("Pretrained model loaded")

    # mCSV=LiTS_Data.read_in_csv(mPath.CSVPath+"predict.csv")
    predict_mode = 4  # 1-预测一个patch, #2-预测一个layer, #3-预测一个nii
    if predict_mode==1:
        img = cv2.imread('E:/WorkSpace/Python/Data/Data_LiTS/volume/volume-121/252-6.jpg')[:, :, 0]
        img = img / 255
        mask=predict_patch(net,img,Use_GPU)
        pass
    if predict_mode==2:
        img=cv2.imread('E:/WorkSpace/Python/Data/Data_LiTS/volume/volume-0/58.jpg')[:,:,0]
        img = img / 255
        samples=predict_layer(net,img,Use_GPU)
        pass
    if predict_mode==3:
        img, img_header = load('E:/WorkSpace/Python/Data/Data_LiTS/Nii/volume-122.nii')
        nii=predict_nii(net,img,Use_GPU,'volume-122')
        pass
    if predict_mode==4:
        for i in range(10):
            name='volume-'+str(i+121)
            img,img_header=load(mPath.DataPath_Nii+name+'.nii')
            nii=predict_nii(net,img,Use_GPU,name)