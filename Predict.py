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

    # preds = ut.Merge_Patches(ret_samples, [512, 512], 5)
    preds = ut.Merge_Patches_And(ret_samples, [512, 512], 5)
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

def predict_layer_multi(net,layer_input,gpu):
    net.eval()

    ret_samples=np.zeros((layer_input.shape[0],layer_input.shape[1],layer_input.shape[3]),dtype='float64')
    for patch in range(layer_input.shape[3]):
        temp=layer_input[:,:,:,patch]
        input=np.zeros((layer_input.shape[2],layer_input.shape[0],layer_input.shape[1]))
        for i in range(temp.shape[2]):
            input[i,:,:]=temp[:,:,i]
            pass
        # cv2.imshow('1', input[1, :, :])
        # cv2.waitKey(0)
        input=input[np.newaxis,:,:,:]

        input=torch.from_numpy(input)
        input=input.type(torch.FloatTensor)

        if gpu:
            input = input.cuda(0)
            pass
        # 只有训练阶段才追踪历史
        with torch.set_grad_enabled(False):

            output = net(input)
            _, preds = torch.max(output, 1)
            ret_samples[:, :, patch] = preds[0, :, :].cpu().numpy()
            # cv2.imshow('2', ret_samples[:,:,patch])
            # cv2.waitKey(0)
            # ut.plot_img(preds[0, :, :], mPath.DataPath_Volume_Predict + "output"+str(patch)+".jpg", 'Output', 2)
            pass
        pass
    preds = ut.Merge_Patches_And(ret_samples, [512, 512], 5)
    preds[preds>0]=2
    # cv2.imshow('2', preds*127)
    # cv2.waitKey(0)
    return preds

def predict_nii_multi(net,img1,img2,gpu,name,is_test=False):
    net.eval()

    img1 = np.array(img1, dtype='float64')

    img1[img1 < -200] = -200
    img1[img1 > 250] = 250
    img1 = ((img1 + 200)*255 // 450)
    img1 = np.array(img1, dtype='uint8')

    if not is_test:
        img2 = np.array(img2, dtype='uint8')

        for i in range(img1.shape[2]):
            img1[:,:,i]=np.flip(img1[:,:,i],0)
            img1[:, :, i] = np.rot90(img1[:, :, i])
            img1[:, :, i] = np.rot90(img1[:, :, i])
            img1[:, :, i] = np.rot90(img1[:, :, i])

        startposition, endposition = pp.getRangImageDepth(img2)

        sub_srcimages = pp.make_multi_patch(img1, (128,128), 5, 3, startposition, endposition)/255

        layers=img1.shape[2]
        save_nii = np.zeros((img1.shape[0], img1.shape[1], layers), dtype='uint8')

        for ind in range(startposition,endposition+1,1):
            # ind=startposition+1
            layer_input=sub_srcimages[:,:,:,(ind-startposition)*25:(ind-startposition)*25+25]

            im= predict_layer_multi(net, layer_input, gpu)
            im=np.flip(im,0)
            im = np.rot90(im)
            im = np.rot90(im)
            im = np.rot90(im)
            # cv2.imshow('2', im*127)
            # cv2.waitKey(0)

            save_nii[:, :, ind]=im
            print("Predicting {}-{}".format(name, ind))
            pass
        pass
    else:
        for i in range(img1.shape[2]):
            img1[:, :, i] = np.flip(img1[:, :, i], 0)
            img1[:, :, i] = np.rot90(img1[:, :, i])
            img1[:, :, i] = np.rot90(img1[:, :, i])
            img1[:, :, i] = np.rot90(img1[:, :, i])

        startposition=1
        endposition=img1.shape[2]-2

        sub_srcimages = pp.make_multi_patch(img1, (128, 128), 5, 3, startposition, endposition) / 255

        layers = img1.shape[2]
        save_nii = np.zeros((img1.shape[0], img1.shape[1], layers), dtype='uint8')

        for ind in range(startposition, endposition + 1, 1):
            # ind=startposition+1
            layer_input = sub_srcimages[:, :, :, (ind - startposition) * 25:(ind - startposition) * 25 + 25]

            im = predict_layer_multi(net, layer_input, gpu)
            im = np.flip(im, 0)
            im = np.rot90(im)
            im = np.rot90(im)
            im = np.rot90(im)
            # cv2.imshow('2', im*127)
            # cv2.waitKey(0)

            save_nii[:, :, ind] = im
            print("Predicting {}-{}".format(name, ind))
            pass
        pass
    save_nii=np.array(save_nii,dtype='uint8')
    new_img = nib.Nifti1Image(save_nii, affine=np.eye(4))
    nib.save(new_img, mPath.DataPath_Volume_Predict + name + '.nii')

if __name__=='__main__':
    # net=UNet_Yading(n_channels=1,n_classes=Output_Class)
    net=torch.load(mPath.DataPath_Net_Test)
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
    predict_mode = 5  # 1-预测一个patch, #2-预测一个layer, #3-预测一个nii
    if predict_mode==1:
        img = cv2.imread('F:\Workspace\python\Data\Data_LiTS/volume/volume-121-1736/0.jpg')[:, :, 0]
        img = img / 255
        mask=predict_patch(net,img,Use_GPU)
        pass
    if predict_mode==2:
        img=cv2.imread('F:\Workspace\python\Data\Data_LiTS/volume/volume-121-1736/0.jpg')[:,:,0]
        img = img / 255
        samples=predict_layer(net,img,Use_GPU)
        pass
    if predict_mode==3:
        img, img_header = load('F:/WorkSpace/Python/Data/Data_LiTS/Nii/volume-121.nii')
        tru,tru_header=load('F:/WorkSpace/Python/Data/Data_LiTS/Nii/segmentation-121.nii')
        # nii=predict_nii(net,img,Use_GPU,'volume-122')
        nii=predict_nii_multi(net,img,tru,Use_GPU,'volume-121')
        pass
    if predict_mode==4:
        for i in range(10):
            name1='volume-'+str(i+121)
            name2 = 'segmentation-' + str(i + 121)
            img,img_header=load(mPath.DataPath_Nii+name1+'.nii')
            tru,tru_header=load(mPath.DataPath_Nii+name2+'.nii')
            # nii=predict_nii(net,img,Use_GPU,name1)
            nii=predict_nii_multi(net,img,tru,Use_GPU,'test-segmentation-'+str(i)+'.nii',False)
            pass
        pass
    if predict_mode==5:#for test only
        for i in range(68,70,1):
            name1='test-volume-'+str(i)+'.nii'
            name2='test-segmentation-'+str(i)+'.nii'
            img, img_header = load(mPath.DataPath_Nii_Predict + name1)
            nii = predict_nii_multi(net, img, img, Use_GPU, 'test-segmentation-' + str(i),True)