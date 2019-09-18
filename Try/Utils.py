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
        img= np.array(img,dtype='uint8')
        cv2.imwrite(path, img)
        pass
    elif type=='Mask':
        img = img.cpu()
        img = img.detach().numpy()
        data=OnehotToLabel(img,classes)*127
        cv2.imwrite(path, data)
        pass
    pass

def save_img(img,path):
    cv2.imwrite(path, img*127)

def dice_cofficient(truth,output,layer=1):
    truth=truth.cpu()
    truth=truth.numpy()

    output=output.cpu()
    output=output.numpy()

    the_other=3-layer
    truth[truth==the_other]=0
    output[output==the_other]=0

    s=np.sum(truth)
    if(s==0):
        return 1

    d1=np.sum(truth[output==layer])
    d2=np.sum(truth[truth==layer])
    d3=np.sum(output[output==layer])
    d4=np.sum(output[truth==layer])
    dice=d1*2/(d2+d3)
    if not (dice>=0 and dice<=1):
        print(d4)
    return dice

def Merge_Patches(samples,size,patch):
    block_width=samples.shape[0]
    block_height=samples.shape[1]
    width=size[0]
    height=size[1]
    stride_width = (width - block_width) // (patch - 1)
    stride_height = (height - block_height) // (patch - 1)

    ret_samples = np.zeros(shape=(width,height), dtype=np.uint8)
    index = 0
    for x in range(0, width - block_width + 1, stride_width):
        for y in range(0, height - block_height + 1, stride_height):
            ret_samples[x:x + block_width, y:y + block_height] = ret_samples[x:x + block_width, y:y + block_height]+samples[:,:,index]
            index = index + 1
            pass
        pass
    ret_samples[ret_samples>=1]=1

    return ret_samples

def Merge_Patches_And(samples,size,patch):
    block_width=samples.shape[0]
    block_height=samples.shape[1]
    width=size[0]
    height=size[1]
    stride_width = (width - block_width) // (patch - 1)
    stride_height = (height - block_height) // (patch - 1)

    ret_samples=np.ones(shape=(width,height),dtype=np.uint8)
    index = 0
    for x in range(0, width - block_width + 1, stride_width):
        for y in range(0, height - block_height + 1, stride_height):
            ret_samples[x:x + block_width, y:y + block_height] = ret_samples[x:x + block_width, y:y + block_height]*samples[:,:,index]
            index = index + 1
            pass
        pass
    ret_samples[ret_samples>=1]=2

    return ret_samples