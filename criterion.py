import torch
import numpy as np

class criterion():
    
    def softmax(self,input):
        
        inp = input.clone()
        inp -= torch.max(inp,1,True)[0]
        exp_op = torch.exp(inp)
        norm_exp = exp_op/exp_op.sum(1,True)
        
        return norm_exp    
        
    def forward(self,input,target):

        log_loss = -torch.log(self.softmax(input))
        index = torch.arange(len(target))
        avg_loss = log_loss[index,target]
        
        return avg_loss
    
    def backward(self,input,target):

        onhot = torch.zeros_like(input)
        index = torch.arange(len(target))
        onhot[index,target]=1
        
        gradLoss = self.softmax(input) - onhot
        
        return gradLoss
            

    
