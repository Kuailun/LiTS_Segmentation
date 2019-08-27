import os

from Try import mPath

from medpy.io import load, save
import os.path
import numpy as np
import csv
import cv2
import random


def checkdirs():
    if not os.path.exists(mPath.DataPath_Volume):
        os.mkdir(mPath.DataPath_Volume)
    if not os.path.exists(mPath.DataPath_Mask):
        os.mkdir(mPath.DataPath_Mask)
        pass
    pass

def preprocessing_filter(nii,volume,mask,resize,rate=0.1):
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

        layers = img1.shape[2]
        black=[]
        for j in range(layers):
            if(np.max(img2[:,:,j])==0):
                black.append(j)
                continue
            else:
                im1 = np.array(img1[:, :, j])
                im2 = np.array(img2[:, :, j])

                im1 = cv2.resize(im1, resize)
                im2 = cv2.resize(im2, resize)

                save(im1, volume + 'volume-' + str(i) + '/' + str(j) + '.jpg')
                save(im2, mask + 'segmentation-' + str(i) + '/' + str(j) + '.jpg')
                print("Saving image " + str(i) + " "+ str(j))
            pass

        # Add black images
        all_num=int(rate*layers)
        if(all_num>len(black)):
            all_num=len(black)
            pass

        # shuffle
        random.shuffle(black)

        for k in range(all_num):
            im1 = np.array(img1[:, :, black[k]])
            im2 = np.array(img2[:, :, black[k]])

            im1 = cv2.resize(im1, resize)
            im2 = cv2.resize(im2, resize)

            save(im1, volume + 'volume-' + str(i) + '/' + str(black[k]) + '.jpg')
            save(im2, mask + 'segmentation-' + str(i) + '/' + str(black[k]) + '.jpg')
            print("Saving image " + str(i) + " " + str(black[k]))
        pass




def preprocessing(image_path, type,save_folder):
    filelist = os.listdir(image_path)
    for i in range(len(filelist)):
        fileName=type+'-'+str(i)+".nii"
        folderName=type+'-'+str(i)+"/"

        #convert file:
        if not os.path.exists(save_folder+folderName):
            os.mkdir(save_folder+folderName)
            pass
        img, img_header = load(image_path + fileName)
        if(type=='segmentation'):
            img=np.array(img,dtype='uint8')
            pass
        elif(type=='volume'):
            img[img < -200] = -200
            img[img > 250] = 250
            img = ((img + 200) * 255 // 450)
            img = np.array(img, dtype='uint8')
            pass
        layers=img.shape[2]
        for j in range(layers):
            save(img[:, :, j], save_folder + folderName+str(j)+".jpg")
            print("Saving image " + save_folder + folderName+str(j)+".jpg")
            pass
        pass
    pass


def generate_livertxt(image_path, save_folder):
    if not os.path.exists("data/" + save_folder):
        os.mkdir("data/" + save_folder)

    # Generate Livertxt
    if not os.path.exists("data/" + save_folder + 'LiverPixels'):
        os.mkdir("data/" + save_folder + 'LiverPixels')

    for i in range(0, 131):
        livertumor, header = load(image_path + 'segmentation-' + str(i) + '.nii')
        f = open('data/' + save_folder + '/LiverPixels/liver_' + str(i) + '.txt', 'w')
        index = np.where(livertumor == 1)
        x = index[0]
        y = index[1]
        z = index[2]
        np.savetxt(f, np.c_[x, y, z], fmt="%d")

        f.write("\n")
        f.close()


def generate_tumortxt(image_path, save_folder):
    if not os.path.exists("data/" + save_folder):
        os.mkdir("data/" + save_folder)

    # Generate Livertxt
    if not os.path.exists("data/" + save_folder + 'TumorPixels'):
        os.mkdir("data/" + save_folder + 'TumorPixels')

    for i in range(0, 131):
        livertumor, header = load(image_path + 'segmentation-' + str(i) + '.nii')
        f = open("data/" + save_folder + "/TumorPixels/tumor_" + str(i) + '.txt', 'w')
        index = np.where(livertumor == 2)

        x = index[0]
        y = index[1]
        z = index[2]

        np.savetxt(f, np.c_[x, y, z], fmt="%d")

        f.write("\n")
        f.close()


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



checkdirs()

# 输出所有图片
# preprocessing(image_path=mPath.DataPath_Nii, type='volume',save_folder=mPath.DataPath_Volume)
# preprocessing(image_path=mPath.DataPath_Nii, type='segmentation',save_folder=mPath.DataPath_Mask)

# 仅输出带有标记的图片
# preprocessing_filter(mPath.DataPath_Nii,mPath.DataPath_Volume,mPath.DataPath_Mask,resize=(128,128),rate=0.1)
generate_csv(mPath.DataPath_Mask, mPath.DataPath_Volume, mPath.CSVPath)

#
# a=[[1,0,0],[0,2,0],[1,2,1]]
# b=[[0,0,0],[0,1,0],[1,2,1]]
# a=np.array(a)
# b=np.array(b)
# currnet=1
# caadsf=3-currnet
#
# a[a==caadsf]=0
# b[b==caadsf]=0
#
# up=np.sum(a[b==1])
# down=np.sum(a[a==1])+np.sum(b[b==1])
# print(up*2/down)