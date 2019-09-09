import os

import mPath

from medpy.io import load, save
import os.path
import numpy as np
import csv
import cv2
import random

from Try import Utils as ut
ut.CheckDirectory(mPath.DataPath_Volume_Predict)

def getRangImageDepth(image):
    """
    :param image:
    :return:rangofimage depth
    """
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[2]):
        notzeroflag = np.max(image[:,:,z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

def preprocessing_filter(nii,volume,mask,resize,rate,slices):
    for i in range(131):
        volumeName='volume-'+str(i)+'.nii'
        segmentName='segmentation-'+str(i)+'.nii'

        if not os.path.exists(volume+'volume-'+str(i)):
            os.mkdir(volume+'volume-'+str(i))
            pass
        if not os.path.exists(mask+'segmentation-'+str(i)):
            os.mkdir(mask+'segmentation-'+str(i))
            pass

        img1, img_header1 = load(nii + volumeName)
        img2, img_header2 = load(nii + segmentName)

        img1[img1<-200]=-200
        img1[img1>250]=250
        img1=((img1+200)*255//450)
        img1=np.array(img1,dtype='uint8')
        img2=np.array(img2,dtype='uint8')

        startposition,endposition=getRangImageDepth(img2)

        layers = img1.shape[2]

        realstart=max(int(startposition*(1-rate))-slices//2,0)
        realend=min(int((layers-endposition)*rate+endposition)+slices//2,layers)


        for j in range(realend-realstart):
            index=j+realstart

            im1 = np.array(img1[:, :, index])
            im2 = np.array(img2[:, :, index])

            im1 = cv2.resize(im1, resize)
            im2 = cv2.resize(im2, resize)

            save(im1, volume + 'volume-' + str(i) + '/' + str(index) + '.jpg')
            save(im2, mask + 'segmentation-' + str(i) + '/' + str(index) + '.jpg')
            print("Saving image " + str(i) + " "+ str(index))
            pass
        pass
    pass

def make_patch(im,patch,resize):
    width=im.shape[0]
    height=im.shape[1]
    block_width=resize[0]
    block_height=resize[1]
    stride_width=(width-block_width)//(patch-1)
    stride_height=(height-block_height)//(patch-1)

    samples=np.empty(shape=(block_width,block_height,patch*patch),dtype='float64')
    index=0
    for x in range(0,width-block_width+1,stride_width):
        for y in range(0,height-block_height+1,stride_height):
            samples[:,:,index]=im[x:x+block_width,y:y+block_height]
            index=index+1
            pass
        pass

    return samples

def preprocessing_patch(nii,volume,mask,resize,patch,randomize):
    for i in range(131):
        volumeName='volume-'+str(i)+'.nii'
        segmentName='segmentation-'+str(i)+'.nii'

        if not os.path.exists(volume+'volume-'+str(i)):
            os.mkdir(volume+'volume-'+str(i))
            pass
        if not os.path.exists(mask+'segmentation-'+str(i)):
            os.mkdir(mask+'segmentation-'+str(i))
            pass

        img1, img_header1 = load(nii + volumeName)
        img2, img_header2 = load(nii + segmentName)

        img1=np.array(img1,dtype='float64')

        img1[img1<-200]=-200
        img1[img1>250]=250
        img1=((img1+200)*255//450)
        img1=np.array(img1,dtype='uint8')
        img2=np.array(img2,dtype='uint8')

        startposition,endposition=getRangImageDepth(img2)

        for j in range(endposition-startposition):
            index=j+startposition
            mMax=np.max(img2[:,:,index])

            if not mMax== 2:
                continue
                pass

            im1 = np.array(img1[:, :, index])
            im2 = np.array(img2[:, :, index])

            sub_volume=make_patch(im1,patch,resize)
            sub_segment=make_patch(im2,patch,resize)

            for ind in range(patch*patch):
                mMaxP=np.max(sub_segment[:,:,ind])
                if not mMaxP==2:
                    # n=random.randint(1,100)
                    # if(n<rate*100):
                    #     save(sub_volume[:, :, ind], volume + 'volume-' + str(i) + '/' + str(index) + "-" + str(ind) + '.jpg')
                    #     save(sub_segment[:, :, ind], mask + 'segmentation-' + str(i) + '/' + str(index) + "-" + str(ind) + '.jpg')
                    continue
                    pass
                sub_volume=np.array(sub_volume,dtype='uint8')
                sub_segment=np.array(sub_segment,dtype='uint8')
                save(sub_volume[:,:,ind], volume + 'volume-' + str(i) + '/' + str(index)+"-"+str(ind) + '.jpg')
                save(sub_segment[:,:,ind], mask + 'segmentation-' + str(i) + '/' + str(index)+"-"+str(ind) + '.jpg')
                print("Saving image " + str(i) + " " + str(index)+"-"+str(ind))

def make_multi_patch(im,resize,patch,layer,startposition,endposition):
    """
    make number patch
    :param image:[depth,512,512]
    :param patch_block: such as[64,128,128]
    :return:[samples,64,128,128]
    expand the dimension z range the subimage:[startpostion-blockz//2:endpostion+blockz//2,:,:]
    """
    width = im.shape[0]
    height = im.shape[1]
    block_width = resize[0]
    block_height = resize[1]
    stride_width = (width - block_width) // (patch - 1)
    stride_height = (height - block_height) // (patch - 1)
    pre_start=layer//2
    realstart=max(pre_start,startposition)
    realend=min(endposition,im.shape[2]-1-pre_start)


    samples = np.empty(shape=(block_width, block_height, layer,patch * patch*(realend-realstart+1)), dtype='float64')
    index = 0
    for lay in range(realstart,realend+1,1):
        for x in range(0, width - block_width + 1, stride_width):
            for y in range(0, height - block_height + 1, stride_height):
                samples[:, :, :,index] = im[x:x + block_width, y:y + block_height,lay-pre_start:lay+pre_start+1]
                index = index + 1
                pass
            pass

    return samples
def preprocessing_multi_patch(nii,volume,mask,resize,patch,layer):
    for i in range(131):
        volumeName='volume-'+str(i)+'.nii'
        segmentName='segmentation-'+str(i)+'.nii'

        img1, img_header1 = load(nii + volumeName)
        img2, img_header2 = load(nii + segmentName)

        img1=np.array(img1,dtype='float64')

        img1[img1<-200]=-200
        img1[img1>250]=250
        img1=((img1+200)*255//450)
        img1=np.array(img1,dtype='uint8')
        img2=np.array(img2,dtype='uint8')

        startposition,endposition=getRangImageDepth(img2)

        sub_srcimages=make_multi_patch(img1,resize,patch,layer,startposition,endposition)
        sub_truthimage=make_multi_patch(img2,resize,patch,layer,startposition,endposition)

        for ind in range(sub_truthimage.shape[3]):
            mMaxP = np.max(sub_truthimage[:, :, layer//2+1,ind])
            if not mMaxP == 2:
                # n=random.randint(1,100)
                # if(n<rate*100):
                #     save(sub_volume[:, :, ind], volume + 'volume-' + str(i) + '/' + str(index) + "-" + str(ind) + '.jpg')
                #     save(sub_segment[:, :, ind], mask + 'segmentation-' + str(i) + '/' + str(index) + "-" + str(ind) + '.jpg')
                continue
                pass
            sub_srcimages = np.array(sub_srcimages, dtype='uint8')
            sub_truthimage = np.array(sub_truthimage, dtype='uint8')

            for lay in range(layer):
                if not os.path.exists(volume + 'volume-' + str(i)+"-"+str(ind)):
                    os.mkdir(volume + 'volume-' + str(i)+"-"+str(ind))
                    pass

                save(sub_srcimages[:, :, lay,ind], volume + 'volume-' + str(i)+"-"+str(ind) + '/' + str(lay)+ '.jpg')
            if not os.path.exists(mask + 'segmentation-' + str(i)+"-"+str(ind)):
                os.mkdir(mask + 'segmentation-' + str(i)+"-"+str(ind))
                pass
            save(sub_truthimage[:, :,layer//2, ind], mask + 'segmentation-' + str(i)+"-"+str(ind) + '/' + str(lay)+ '.jpg')
            print("Saving image " + str(i) + " " + str(ind))



def generate_csv(mask_path,volume_path, save_folder,mode,shuffle=False):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        pass
    if mode=='train':
        fileNam='data.csv'
    else:
        fileNam='predict.csv'
    maskList=os.listdir(mask_path)
    volumeList=os.listdir(volume_path)
    if(len(maskList)==len(volumeList)):
        pass
    else:
        raise("2 data directories is not agreed")

    csv_content=[]

    for i in range(len(maskList)):
        maskDir = "segmentation-"+str(i)+"/"
        volumeDir = "volume-" + str(i)+"/"
        maskDirList=os.listdir(mask_path+maskDir)
        volumeDirList = os.listdir(volume_path + volumeDir)
        if (len(maskDirList) == len(volumeDirList)):
            pass
        else:
            raise ("{0} and {1} data directories is not agreed".format(maskDir,volumeDir))
        for j in range(len(maskDirList)):
            if(maskDirList[j] in volumeDirList):
                fileName=maskDirList[j]
                csv_content.append((volume_path+volumeDir+fileName,mask_path+maskDir+fileName))
            pass
        pass

    if mode=='train' and shuffle:
        csv_array=np.array(csv_content)
        np.random.shuffle(csv_array)
        csv_content=csv_array.tolist()

    headers=['volume_path','mask_path']
    with open(save_folder+fileNam,'w',newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_content)
        f.close()
        pass
    pass

def generate_multi_csv(mask_path,volume_path, save_folder,shuffle=False):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        pass
    fileNam='data.csv'
    maskList=os.listdir(mask_path)
    volumeList=os.listdir(volume_path)
    if(len(maskList)==len(volumeList)):
        pass
    else:
        raise("2 data directories is not agreed")

    csv_content=[]
    for i in range(len(maskList)):
        maskDir = maskList[i]
        volumeDir = volumeList[i]
        maskDirList=os.listdir(mask_path+maskDir)
        volumeDirList = os.listdir(volume_path + volumeDir)

        temp = []
        for j in range(len(volumeDirList)):
            temp.append(volume_path+volumeDir+'/'+volumeDirList[j])
        temp.append(mask_path+maskDir+'/'+maskDirList[0])
        csv_content.append(temp)

    if shuffle:
        csv_array=np.array(csv_content)
        np.random.shuffle(csv_array)
        csv_content=csv_array.tolist()

    headers=[]
    for i in range(len(volumeDirList)):
        headers.append('volume_path')
        pass
    headers.append('mask_path')

    with open(save_folder+fileNam,'w',newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_content)
        f.close()
        pass
    pass

def generate_csv_nii(volume_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        pass
    volumeList=os.listdir(volume_path)

    csv_content=[]

    for j in range(len(volumeList)):
        csv_content.append([volume_path+volumeList[j]])
    pass
    headers = ['volume_path']
    with open(save_folder+'predict.csv','w',newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_content)
        f.close()
        pass
    pass

if __name__=='__main__':

    ut.CheckDirectory(mPath.DataPath_Volume)
    ut.CheckDirectory(mPath.DataPath_Mask)
    # 生成完整图
    # preprocessing_filter(mPath.DataPath_Nii, mPath.DataPath_Volume, mPath.DataPath_Mask, resize=(512, 512), rate=0.1, slices=3)

    # 生成单层patch图
    # preprocessing_patch(mPath.DataPath_Nii, mPath.DataPath_Volume, mPath.DataPath_Mask, resize=(128, 128), patch=5,randomize=True)
    # generate_csv(mPath.DataPath_Mask, mPath.DataPath_Volume, mPath.CSVPath, "train", shuffle=False)

    # 生成多层patch图
    # preprocessing_multi_patch(mPath.DataPath_Nii,mPath.DataPath_Volume, mPath.DataPath_Mask, resize=(128, 128), patch=5,layer=3)
    generate_multi_csv(mPath.DataPath_Mask, mPath.DataPath_Volume, mPath.CSVPath, shuffle=False)



