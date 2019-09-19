from medpy.io import load, save
import nibabel as nib
import mPath
import numpy as np
import cv2

for i in range(0,70,1):
    name1='test-liver-'+str(i)+'.nii'
    name2='test-segmentation-'+str(i)+'.nii'
    img1, img1_header = load(mPath.DataPath_Volume_Predict + name1)
    img2, img2_header = load(mPath.DataPath_Volume_Predict + name2)
    layer=img1.shape[2]

    output=np.zeros_like(img1)

    for j in range(layer):
        layer1=img1[:,:,j]
        layer2=img2[:,:,j]
        # cv2.imshow('',layer1*255)
        # cv2.waitKey(0)
        layer2=layer2*layer1
        layer2=layer2*5
        layer1=layer1+layer2
        layer1[layer1>1]=2
        output[:,:,j]=layer1
        print('Merging {}-{}',i,j)
        pass

    output = np.array(output, dtype='uint8')
    new_img = nib.Nifti1Image(output, affine=np.eye(4))
    nib.save(new_img, mPath.DataPath_Volume_Predict + name2)