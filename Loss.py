import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
        pass

    def forward(self,input,target):
        N=target.size(0)
        smooth=1

        input_flat=input.view(N,-1)
        target_flat=target.view(N,-1)

        intersection=input_flat*target_flat

        loss=2*(intersection.sum(1)+smooth)/(input_flat.sum(1)+target_flat.sum(1)+smooth)
        loss=1-loss.sum()/N

        return loss
    pass

class MultclassDiceLoss(nn.Module):
    def __init__(self):
        super(MultclassDiceLoss,self).__init__()
        pass

    def forward(self, input,target,weights=None):
        C=target.shape[1]

        dice=DiceLoss()
        totalLoss=0

        for i in range(C):
            diceLoss=dice(input[:,i],target[:,i])
            if weights is not None:
                diceLoss=diceLoss*weights[i]
                pass
            totalLoss=totalLoss+diceLoss
            pass
        return totalLoss