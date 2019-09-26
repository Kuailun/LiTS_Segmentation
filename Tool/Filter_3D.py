import os
from medpy.io import load, save
import nibabel as nib
import numpy as np
import cv2

def dice_cofficient(truth,output,layer=1):
    s=np.sum(truth)
    p=np.sum(output)
    if(s==0 and p==0):
        return 1

    d1=np.sum(truth[output==layer])
    d2=np.sum(truth[truth==layer])
    d3=np.sum(output[output==layer])
    d4=np.sum(output[truth==layer])
    dice=d1*2/(d2+d3)
    if not (dice>=0 and dice<=1):
        print(d4)
    return dice

def GetMaxLayer(img):
    max_layer=-1
    max_index=-1
    for i in range(img.shape[2]):
        layer_sum=np.sum(img[:,:,i])
        if(layer_sum>max_layer):
            max_layer=layer_sum
            max_index=i
            pass
        pass
    return max_index

def Reserve(img,index,step):
    '''
    仅保留非全黑帧
    :param img:
    :param index:
    :param step:
    :return:
    '''
    if step==-1:
        stop=0
    elif step==1:
        stop=img.shape[2]-1
        pass

    i=index
    dice_list=[]
    #根据dice score找到大致层数分布，并删除层数之外的所有像素，连续4个1即判定为终止
    while not i==stop:
        dice_score=dice_cofficient(img[:,:,i],img[:,:,i+step])
        # print("Layer {0} and {1}: {2}".format(i+1,i+step+1,dice_score))
        dice_list.append(dice_score)
        i=i+step
        pass

    consecutive=0
    pos=-1
    for i in range(len(dice_list)):
        if(consecutive==4):
            layer=index+step*i
            img[:,:,layer]=img[:,:,layer]*0
            # print(layer)
            pass
        else:
            if dice_list[i]==1 or dice_list[i]<0.5:
                consecutive=consecutive+1
                if consecutive==4:
                    pos=i
            else:
                consecutive=0
                pass
            pass
        pass
    img[:,:,stop]=img[:,:,stop]*0
    if pos==-1:
        pos=len(dice_list)-1
        pass

    return img,pos

def Consecutive(img,index,step,pos):

    # 连通域排除
    i=index
    final=False
    while not i==index+step*pos:

        if not final:
            truth=img[:,:,i]
            truth[truth>0]=1

            temp=img[:,:,i+step]
            _,labels=cv2.connectedComponents(temp)
            truth=truth*labels
            if i==303:
                print('here')
                # img = np.array(labels, dtype='uint8')
                # img=np.flip(img,0)
                # img=np.rot90(img)
                # img = np.rot90(img)
                # img = np.rot90(img)
                # cv2.imshow('1', img*50)
                # cv2.waitKey(0)

            truth=np.unique(truth)

            if truth.max()>0:

                # 获得连通域大小列表
                area_list=[]
                for j in range(1,truth.max()+1,1):
                    area_list.append(np.sum(temp[labels==j]))
                    pass
                area_max=max(area_list)

                new_truth=[]
                for j in range(1,truth.max()+1,1):
                    if area_list[j-1]>=40 and area_list[j-1]/area_max>0.2:
                        new_truth.append(j)
                        pass
                    pass

                mmax=labels.max()
                for j in range(1,mmax+1,1):
                    if j not in new_truth:
                        temp[labels==j]=0
                        pass
                    pass
                temp[temp>0]=1
            else:
                img[:,:,i]=img[:,:,i]*0
                final=True
                pass
        else:
            img[:,:,i]=img[:,:,i]*0

        i=i+step
        # print(i)
    return img

for i in range(0,70,1):
    output_name='test-segmentation-'+str(i)+'.nii'
    volume_path='F:/Workspace/python/Data/Data_LiTS/volume_predict/'
    img,img_header=load(volume_path+output_name)
    img=np.array(img,dtype='uint8')
    max_index=GetMaxLayer(img)
    print(i)
    img,pos_f=Reserve(img,max_index,-1)
    img,pos_a=Reserve(img,max_index,1)

    img = Consecutive(img, max_index, 1, pos_f)
    img = Consecutive(img, max_index, -1, pos_f)


    output=np.array(img,dtype='uint8')
    output = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(output, volume_path +'test-segmentation--'+str(i)+'.nii')