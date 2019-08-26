from Model.mModel import UNet
import os
import torch
import mPath
import Utils as ut

GPU_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_DEVICES
Use_GPU=torch.cuda.is_available()
Output_Class=3

def predict_net():
    pass





if __name__=='__main__':
    net=UNet(n_channels=1,n_classes=Output_Class)

    ut.CheckDirectory(mPath.DataPath_Net_CheckPoint)
    if Use_GPU:
        net.cuda()
        net.load_state_dict(torch.load(mPath.DataPath_Net_Normal))
        pass
    else:
        net.load_state_dict(torch.load(mPath.DataPath_Net_Normal,map_location='cpu'))

    try:
        predict_net()
        train_net(net,lr=learning_rate,epochs=Train_Epochs,batch_size=Train_Batch_Size,val_percent=Validation_Percent,save_cp=Save_CheckPoint,gpu=Use_GPU,classes=Output_Class)

    except KeyboardInterrupt:
        ut.CheckDirectory(mPath.DataPath_Net_CheckPoint)
        torch.save(net.state_dict(),mPath.DataPath_Net_Interrupt)
