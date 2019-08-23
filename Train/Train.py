from Datasets import LiTS_Data
from Model.mModel import UNet
from torch.utils.data import DataLoader
from torch import optim,nn
import torch
import mPath
import torch
import cv2
import pandas as pd
from Utils import Utils as ut

Use_GPU=torch.cuda.is_available()
Output_Class=3
Train_Epochs=5
Train_Batch_Size=1
Validation_Percent=0.05
Save_CheckPoint=True

print(Use_GPU)

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.01,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              classes=3):
    optimizer=optim.SGD(net.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=0.0005)
    optimizer.zero_grad()
    criterion=nn.BCELoss()

    mTrain,mValid = LiTS_Data.split_to_train_val(mPath.CSVPath + "data.csv",val_percent)

    mTrainDataset=LiTS_Data.mDataset(mTrain,classes=classes)
    mValDataset=LiTS_Data.mDataset(mValid,classes=classes)

    mTrainDataloader=DataLoader(dataset=mTrainDataset,batch_size=batch_size,shuffle=False)
    mValDataloader = DataLoader(dataset=mValDataset, batch_size=batch_size, shuffle=False)


    best_acc=0.0

    # Begin training
    for epoch in range(epochs):
        print()
        print()
        print('-' * 10)
        print("Starting epoch {}/{}".format(epoch+1,epochs))


        for phase in ['train','val']:
            print()
            if(phase=='train'):
                net.train()
                dataset=mTrainDataloader
                dataLength=len(mTrainDataset)
            else:
                net.eval()
                dataset=mValDataloader
                dataLength = len(mValDataset)
                pass

            running_loss=0.0
            running_corrects=0

            index=0
            for i,sample in enumerate(dataset):
                index=index+1
                img = sample['img']
                mask = sample['mask']

                if gpu:
                    img = img.cuda()
                    mask = mask.cuda()
                    pass

                img = img.float()
                mask = mask.float()

                # 清空参数的梯度
                optimizer.zero_grad()

                # 只有训练阶段才追踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    output = net(img)
                    _, preds = torch.max(output, 1)

                    preds_onehot=torch.zeros_like(mask)
                    for i in range(preds_onehot.shape[1]):
                        preds_onehot[:,i,:,:]=preds==i

                    loss = criterion(output, mask)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        pass

                #  记录loss和准确率
                subloss=loss.item()
                subacc=torch.sum(preds_onehot==mask.data)
                subacc=subacc.float()
                subacc=subacc/img.shape[0]/classes/img.shape[2]/img.shape[3]

                running_loss+=subloss*img.size(0)
                running_corrects+=subacc*img.size(0)

                print('{} Loss:{:.4f} Acc:{:.10f}'.format(index, subloss, subacc))
                pass
            epoch_loss=running_loss/dataLength
            epoch_acc=running_corrects/dataLength

            print('{} Loss:{:.4f} Acc:{:.10f}'.format(phase,epoch_loss,epoch_acc))

            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                ut.CheckDirectory(mPath.DataPath_Net_CheckPoint)
                torch.save(net.state_dict(), mPath.DataPath_Net_Normal)
                pass
            pass
        pass
    pass

if __name__=='__main__':
    net=UNet(n_channels=1,n_classes=Output_Class)

    if Use_GPU:
        net.cuda()
        pass

    try:
        train_net(net,epochs=Train_Epochs,batch_size=Train_Batch_Size,val_percent=Validation_Percent,save_cp=Save_CheckPoint,gpu=Use_GPU,classes=Output_Class)

    except KeyboardInterrupt:
        ut.CheckDirectory(mPath.DataPath_Net_CheckPoint)
        torch.save(net.state_dict(),mPath.DataPath_Net_Interrupt)
