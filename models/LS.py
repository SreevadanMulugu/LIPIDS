import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

class LS(nn.Module):
    def __init__(self):
        super(LS, self).__init__()
        

    def forward(self, x):
        img = x[0]
        light = x[1]
        img = img.permute(0,2,3,1)
        light = light.permute(0,2,3,1)
        img = img.reshape(img.shape[0], img.shape[1], img.shape[2], img.shape[3]//3,3)
        light = light.reshape(light.shape[0], light.shape[1], light.shape[2], light.shape[3]//3,3)
        I = torch.norm(img, dim=-1, keepdim=True)
        I = torch.matmul(light.transpose(3, 4), I)
        L = torch.inverse(torch.matmul(light.transpose(3, 4), light))
        N = torch.matmul(L,I)
        N = N.reshape(N.shape[0],N.shape[1],N.shape[2],N.shape[3])
        e = 1e-10
        N = N/(torch.norm(N, dim=-1, keepdim=True)+e)
        N = N.permute(0,3,1,2)
        
        return N
        
        
        

