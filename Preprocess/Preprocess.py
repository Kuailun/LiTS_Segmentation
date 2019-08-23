import mPath

from medpy.io import load, save
import os
import os.path
import numpy as np
import csv

def checkdirs():
    if not os.path.exists(mPath.DataPath_Volume):
        os.mkdir(mPath.DataPath_Volume)
    if not os.path.exists(mPath.DataPath_Mask):
        os.mkdir(mPath.DataPath_Mask)
def preprecessing(image_path, type,save_folder):
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
            fileName=str(j)+".jpg"
            csv_content.append((volume_path+volumeDir+fileName,mask_path+maskDir+fileName))
            pass
        pass

    headers=['volume_path','mask_path']
    with open(save_folder+'data.csv','w',newline='') as f:
        f_csv=csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(csv_content)
        f.close()
        pass
    pass



checkdirs()
# preprecessing(image_path=mPath.DataPath_Nii, type='volume',save_folder=mPath.DataPath_Volume)
# preprecessing(image_path=mPath.DataPath_Nii, type='segmentation',save_folder=mPath.DataPath_Mask)
generate_csv(mPath.DataPath_Mask,mPath.DataPath_Volume,mPath.CSVPath)