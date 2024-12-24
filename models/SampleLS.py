import torch
import torch.nn as nn
from models.LS import LS


class SampleLS(nn.Module):

    def __init__(self,nin_img,nout_img):
        super(SampleLS,self).__init__()

        self.ls = LS()
        # checkpoint = torch.load('./data/models/PS-FCN_B_S_32.pth.tar')
        # self.ps_fcn.load_state_dict(checkpoint['state_dict'])
        self.nout_img = nout_img
        self.softmax = nn.Softmax(dim=1)

        # li = []
        # for i in range(nout_img):
        #     temp = []
        #     for j in range(nin_img):
        #         if j==i:
        #             temp.append(1)
        #         else:
        #             temp.append(0)
        #     li.append(temp)
        # self.weight = torch.tensor(li,dtype=torch.float).cuda(0)


        self.weight = nn.Parameter(torch.ones(nout_img,nin_img))

    def forward(self,x,epoch):
        epoch = torch.tensor(epoch,dtype=torch.float).cuda(0)
        multiply_by = self.softmax((10*epoch*epoch)*self.weight)
        # print(multiply_by.device)
        # multiply_by = self.weight
        img = x[0]
        img_split = torch.split(img, 3, 1)
        img_stacked = torch.stack(img_split, dim=1)
        img_reshaped = img_stacked.view(img_stacked.shape[0], img_stacked.shape[1],-1)
        # print('***************',img_reshaped.shape, multiply_by.shape)
        img_reshaped = torch.matmul(multiply_by, img_reshaped)
        img_reshaped = img_reshaped.view(img_reshaped.shape[0],self.nout_img*img_stacked.shape[2],img_stacked.shape[3],img_stacked.shape[4])
    
        light = x[1]
        light_split = torch.split(light, 3, 1)
        light_stacked = torch.stack(light_split, dim=1)
        light_reshaped = light_stacked.view(light_stacked.shape[0], light_stacked.shape[1],-1)
        light_reshaped = torch.matmul(multiply_by, light_reshaped)
        light_reshaped = light_reshaped.view(light_stacked.shape[0],self.nout_img*light_stacked.shape[2],light_stacked.shape[3],light_stacked.shape[4])
        ans = self.ls((img_reshaped, light_reshaped))
        return ans
        # return self.ps_fcn(x)