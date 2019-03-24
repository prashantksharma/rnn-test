import torch
import numpy as np
from random import sample

class rnn():
    
    def __init__(self):
        pass
    def forward(self,hprev,input_batch,Bh,Whh,Wxh):

        theta = hprev.matmul(Whh) + input_batch.matmul(Wxh) + Bh
        self.output = torch.tanh(theta)
        return self.output
        
    def backward(self,gradOutput,input_batch,hprev,Whh):
        
        self.gradBh = gradOutput*(1-self.output**2)
        self.gradWhh = hprev.transpose(1,0).matmul(self.gradBh)
        self.gradWxh = input_batch.transpose(1,0).matmul(self.gradBh)
        gradInput = self.gradBh.matmul(Whh.transpose(1,0))
        ''' Dealing with vanishing and Exploding the gradients '''
        norm_gradIn = np.linalg.norm(gradInput)
        norm_gradOt = np.linalg.norm(gradOutput)
       # print("grad_in_norm: "+str(norm_gradIn)+" grad_out_norm: "+str(norm_gradOt))
        gradInput *= ((0.01+norm_gradOt)/(0.01+norm_gradIn))

        return gradInput
        

#hprev = torch.tensor([[1,2,3],[1,2,3],[1,2,3]]).double()
#input_batch = torch.tensor([[2,2,2,2],[3,3,3,3],[4,4,4,4]]).double() 3x4
# batch_size x len_unique
#Whh = torch.tensor([[1,2,3],[1,2,3],[1,2,3]]).double()
#Wxh = torch.tensor([[3,3,3],[3,3,3],[3,3,3],[3,3,3]]).double()
#Bh = torch.tensor([1,2,3]).double()
#gradOutput= torch.tensor([[1,2,3],[1,2,3],[1,2,3]]).double()
