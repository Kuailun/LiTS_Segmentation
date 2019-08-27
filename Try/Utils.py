import os
import cv2
import numpy as np
def CheckDirectory(p):
    """Check whether directory is existed, if not then create"""
    if os.path.exists(p):
        pass
    else:
        os.mkdir(p)
        pass
    pass

def OnehotToLabel(img,classes):
    if(classes==2):
        data=img[1,:,:]
    else:
        data=img[1,:,:]+2*img[2,:,:]
    return data

def plot_img(img,path,type,classes):
    '''Save img to directory'''
    if type=="Input":
        img=img.cpu()
        img=img.numpy()*255
        cv2.imwrite(path,img)
        pass
    elif type=="Output":
        img = img.cpu()
        img = img.detach().numpy()*127
        cv2.imwrite(path, img)
        pass
    elif type=='Mask':
        img = img.cpu()
        img = img.detach().numpy()
        data=OnehotToLabel(img,classes)*127
        cv2.imwrite(path, data)
        pass
    pass

def dice_cofficient(truth,output,layer=1):
    truth=truth.cpu()
    truth=truth.numpy()

    output=output.cpu()
    output=output.numpy()

    the_other=3-layer
    truth[truth==the_other]=0
    output[output==the_other]=0

    d1=np.sum(truth[output==layer])
    d2=np.sum(truth[truth==layer])
    d3=np.sum(output[output==layer])
    d4=np.sum(output[truth==layer])
    dice=d1*2/(d2+d3)
    if not (dice>=0 and dice<=1):
        print('here')
        print(d4)
    return dice