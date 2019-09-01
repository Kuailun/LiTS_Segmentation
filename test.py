from Datasets import LiTS_Data
from Try.Model.mModel import *
from torch.utils.data import DataLoader
from torch import optim
from Try import Loss, Utils as ut
import torch
from tensorboardX import SummaryWriter
import os
import mPath

GPU_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_DEVICES
Use_GPU=torch.cuda.is_available()
Output_Class=2
Train_Epochs=1
Train_Batch_Size=1
Validation_Percent=0.1
Save_CheckPoint=True
Output_per_epoch=1
learning_rate=0.01
weights=[0,1,0]

writer=SummaryWriter()


print("GPU status {}".format(Use_GPU))

def adjust_learning_rate(optimizer,epoch):
    lr=learning_rate*(0.1**(epoch//500))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
        pass

    writer.add_scalar('learning_rate' , lr,epoch)

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.01,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              classes=3):

    mTrain,mValid = LiTS_Data.split_to_train_val(mPath.CSVPath + "data.csv", val_percent)

    mTrainDataset= LiTS_Data.Dataset_WithLiver(mTrain, classes=classes)

    mTrainDataloader=DataLoader(dataset=mTrainDataset,batch_size=batch_size,shuffle=True)

    for epoch in range(epochs):

        net.eval()
        dataset=mTrainDataloader

        for index,sample in enumerate(dataset):
            img = sample['img']
            mask = sample['mask']

            if gpu:
                img = img.cuda()
                mask = mask.cuda()
                pass

            img = img.float()
            mask = mask.float()

            # 只有训练阶段才追踪历史
            with torch.set_grad_enabled(False):
                output = net(img)
                _, preds = torch.max(output, 1)

                preds_onehot=torch.zeros_like(mask)
                for i in range(preds_onehot.shape[1]):
                    preds_onehot[:,i,:,:]=preds==i
                    pass
            pass

        if epoch%Output_per_epoch==0:
            ut.plot_img(img[0,0,:,:], mPath.DataPath_Log + "Input0-" + str(epoch) + ".jpg", "Input",2)
            ut.plot_img(mask[0, :, :, :], mPath.DataPath_Log + "Mask0-" + str(epoch) + ".jpg", "Mask",2)
            ut.plot_img(preds[0, :, :], mPath.DataPath_Log + "Output0-" + str(epoch) + ".jpg", "Output",2)
            pass
        pass
    writer.close()
    pass

if __name__=='__main__':
    net=UNet_Yading(n_channels=1,n_classes=Output_Class)

    if Use_GPU:
        net.cuda()
        net.load_state_dict(torch.load(mPath.DataPath_Net_Predict))
        pass

    try:
        ut.CheckDirectory(mPath.DataPath_Log)
        train_net(net, lr=learning_rate, epochs=Train_Epochs, batch_size=Train_Batch_Size,val_percent=Validation_Percent, save_cp=Save_CheckPoint, gpu=Use_GPU, classes=Output_Class)

    except KeyboardInterrupt:
        ut.CheckDirectory(mPath.DataPath_Net_CheckPoint)
        torch.save(net.state_dict(), mPath.DataPath_Net_Interrupt)
