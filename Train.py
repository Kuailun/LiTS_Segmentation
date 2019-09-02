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
Train_Epochs=400
Train_Batch_Size=10
Validation_Percent=0.1
Save_CheckPoint=True
Output_per_epoch=5
learning_rate=0.01
weights=[0,1,0]

writer=SummaryWriter()


print("GPU status {}".format(Use_GPU))

def adjust_learning_rate(optimizer,epoch):
    lr=learning_rate*(0.1**(epoch//400))
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
    optimizer=optim.SGD(net.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=0.0005)
    # optimizer=optim.adam(net.parameters(),
    #                      lr=lr,
    #                      momentum=0.9,
    #                      weight_decay=0.0005)
    # optimizer.zero_grad()
    # criterion=nn.BCELoss()
    criterion= Loss.MultclassDiceLoss()

    mTrain,mValid = LiTS_Data.split_to_train_val(mPath.CSVPath + "data.csv", val_percent)

    mTrainDataset= LiTS_Data.Dataset_WithLiver(mTrain, classes=classes)
    mValDataset= LiTS_Data.Dataset_WithLiver(mValid, classes=classes)

    mTrainDataloader=DataLoader(dataset=mTrainDataset,batch_size=batch_size,shuffle=True)
    modeList=[]
    if len(mValid)==0:
        modeList=['train']
    else:
        mValDataloader = DataLoader(dataset=mValDataset, batch_size=batch_size, shuffle=True)
        modeList = ['train', 'val']




    best_acc=0.0
    batch_count=0

    # Begin training
    iter_train=0
    iter_val=0
    for epoch in range(epochs):
        print()
        print()
        print('-' * 10)
        print("Starting epoch {}/{}".format(epoch+1,epochs))

        adjust_learning_rate(optimizer,epoch)

        for phase in modeList:
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
            running_dice=0.0

            for index,sample in enumerate(dataset):
                batch_count=batch_count+1
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
                        pass

                    # loss = criterion(output, mask)
                    loss = criterion(output, mask,weights)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        pass

                #  记录loss和准确率
                subloss=loss.item()
                subaccTotal=torch.sum(preds_onehot==mask.data)

                subaccTotal=subaccTotal.float()
                subaccTotal=subaccTotal/img.shape[0]/classes/img.shape[2]/img.shape[3]

                running_loss+=subloss*img.size(0)
                running_corrects+=subaccTotal*img.size(0)

                mask_ori=mask[:,1,:,:]
                sub_dice=ut.dice_cofficient(mask_ori,preds,1)

                samples=mask.shape[0]
                running_dice=running_dice+sub_dice*samples

                if(phase=='train'):
                    writer.add_scalar('train_loss', subloss, iter_train)
                    writer.add_scalar('train_acc', subaccTotal, iter_train)
                    writer.add_scalar('train_dice', sub_dice, iter_train)
                    iter_train+=1
                elif(phase=='val'):
                    writer.add_scalar('val_loss', subloss, iter_val)
                    writer.add_scalar('val_acc', subaccTotal, iter_val)
                    writer.add_scalar('val_dice', sub_dice, iter_val)
                    iter_val += 1

                print('{} Loss:{:.6f} Acc:{:.10f} Dice:{:.10f}'.format(index, subloss, subaccTotal,sub_dice))
                pass
            if not (dataLength==0):
                epoch_loss=running_loss/dataLength
                epoch_acc=running_corrects/dataLength
                epoch_dice=running_dice/dataLength
            else:
                epoch_loss=0
                epoch_acc=0
                epoch_dice=0

            print('{} Loss:{:.6f} Acc:{:.10f} Dice:{:.10f}'.format(phase,epoch_loss,epoch_acc,epoch_dice))
            # writer.add_scalars('scalar/epoch_data', {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc},
            #                    epoch)


            if epoch%Output_per_epoch==0:
                ut.plot_img(img[0,0,:,:], mPath.DataPath_Log + "Input0-" + str(epoch) + ".jpg", "Input",2)
                ut.plot_img(mask[0, :, :, :], mPath.DataPath_Log + "Mask0-" + str(epoch) + ".jpg", "Mask",2)
                ut.plot_img(preds[0, :, :], mPath.DataPath_Log + "Output0-" + str(epoch) + ".jpg", "Output",2)

            if(phase=='val' and epoch_dice>best_acc):
                best_acc=epoch_dice
                torch.save(net, mPath.DataPath_Net_Normal)
                pass
            pass
        pass
    torch.save(net, mPath.DataPath_Net_Final)
    writer.close()
    pass

if __name__=='__main__':
    net=UNet(n_channels=1,n_classes=Output_Class)
    # dummy_input=torch.rand(Train_Batch_Size,1,256,256)
    # writer.add_graph(net,input_to_model=(dummy_input,))

    if Use_GPU:
        net.cuda()
        pass

    try:
        ut.CheckDirectory(mPath.DataPath_Log)
        train_net(net, lr=learning_rate, epochs=Train_Epochs, batch_size=Train_Batch_Size,val_percent=Validation_Percent, save_cp=Save_CheckPoint, gpu=Use_GPU, classes=Output_Class)

    except KeyboardInterrupt:
        ut.CheckDirectory(mPath.DataPath_Net_CheckPoint)
        torch.save(net.state_dict(), mPath.DataPath_Net_Interrupt)
