import mPath
import os,fnmatch
from medpy.io import load, save
import numpy as np

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
def calculateDice(predicted,original,all=True,withGroundTruth=True):
    img1, img_header1 = load(predicted)
    img2, img_header2 = load(original)
    layers=img1.shape[2]

    overall_dice=0
    ll=0
    for layer in range(layers):
        img1_layer = img1[:, :, layer]
        img2_layer = img2[:, :, layer]

        mask=img2_layer.copy()
        mask[mask==2]=1

        if withGroundTruth:
            img1_layer=img1_layer*mask

        truth=img2_layer.copy()
        truth[truth==2]=1
        # truth[truth==1]=0
        # truth[truth>0]=1

        dice_score=dice_cofficient(truth,img1_layer,1)
        if dice_score>1:
            print('here')
        if all:
            overall_dice=overall_dice+dice_score
            ll=ll+1
            print(str(layer) + '-' + str(dice_score))
        else:
            if(np.sum(truth)>0):
                overall_dice = overall_dice + dice_score
                ll = ll + 1
                print(str(layer) + '-' + str(dice_score))

        pass

    print("Overall dice is {}".format(overall_dice/ll))

    return overall_dice,ll


mode=3
if __name__=='__main__':
    if(mode==1):
        predicted=fnmatch.filter(os.listdir(mPath.DataPath_Volume_Predict),"*.nii")

        original=[mPath.DataPath_Nii+item.replace('-.nii','')+'.nii' for item in predicted]

        predicted = [mPath.DataPath_Volume_Predict + item for item in predicted]

        for i in range(len(predicted)):
            calculateDice(predicted[i], original[i])
            pass
        pass
    if(mode==2):
        calculateDice("F:/WorkSpace/Python/Data/Data_LiTS/volume_predict/volume-121-.nii","F:/WorkSpace/Python/Data/Data_LiTS/Nii/segmentation-121.nii",False)
        pass
    if(mode==3):
        num=10
        pair1 = ['F:/WorkSpace/Python/Data/Data_LiTS/volume_predict/outer-' + str(index + 121) + '.nii' for index in range(num)]
        pair2 = ['F:/WorkSpace/Python/Data/Data_LiTS/Nii/segmentation-' + str(index + 121) + '.nii' for index in range(num)]
        overall_dice=0
        items=0
        average_dice=0
        score_history=[]
        for i in range(len(pair1)):
            temp_o,temp_i=calculateDice(pair1[i],pair2[i],True,False)
            overall_dice=overall_dice+temp_o
            items=items+temp_i
            average_dice=average_dice+temp_o/temp_i
            score_history.append(temp_o/temp_i)
            pass

        print('验证集dice(per frame)='+str(overall_dice/items))
        print('验证集dice(per case)=' + str(average_dice/len(pair1)))
        pass
    print(score_history)