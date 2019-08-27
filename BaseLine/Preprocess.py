import os

from BaseLine import mPath

from medpy.io import load, save
import os.path
import numpy as np
import csv
import cv2
import random

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


def generate_csv(mask_path,volume_path, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        pass
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
    csv_array=np.array(csv_content)
    np.random.shuffle(csv_array)
    csv_content=csv_array.tolist()

    headers=['volume_path','mask_path']
    with open(save_folder+'data.csv','w',newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_content)
        f.close()
        pass
    pass



preprocessing_filter(mPath.DataPath_Nii,mPath.DataPath_Volume,mPath.DataPath_Mask,resize=(128,128),rate=0.1,slices=3)
generate_csv(mPath.DataPath_Mask, mPath.DataPath_Volume, mPath.CSVPath)