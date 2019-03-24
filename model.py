import torch
import numpy as np
from rnn import *

class model():

    def addlayer(self,layer_object):
        
        self.layer.append(layer_object)
    
    def __init__(self,nLayers,H,B,D,isTrain):
        
        self.layer = []
        self.nLayers = nLayers # no_of_layers
        self.hidden_dim = H # hidden_layer_dim
        self.batch_size = B # Batch_Size
        self.input_dim = D # word_vector_size, len_unique
        self.out = 2 # no_of_output_classes
        self.isTrain = isTrain   
        self.Wxh = torch.randn([self.input_dim,self.hidden_dim]).double()*0.1
        self.Whh = torch.randn([self.hidden_dim,self.hidden_dim]).double()*0.1
        self.Bh = torch.randn(self.hidden_dim).double()*0.1
        self.By = torch.randn(self.out).double()*0.1
        self.Why = torch.randn([self.hidden_dim,self.out]).double()*0.1
        
        for t in range(nLayers):
            self.addlayer(rnn())

    def forward(self,input,hprev):
        
        self.h0 = hprev.clone()
        self.hprev = hprev.clone()
        i=0
        for t in range(self.nLayers):
            # print(i)
            
            self.hprev = self.layer[t].forward(self.hprev,input[:,t,:],self.Bh,self.Whh,self.Wxh)
            i=i+1
            
        output = self.hprev.mm(self.Why)
        
        return output,self.hprev
        
    def forward_test(self,inp,hprev):
        

        hp = hprev.clone().view(1,-1)
       # print(hp.shape)
        rec_net = rnn()
        i=0
        for t in range(inp.shape[0]):
            #print(i)
            hp = rec_net.forward(hp,inp[t].view(1,-1),self.Bh,self.Whh,self.Wxh)
            i=i+1
            
        output = hp.mm(self.Why)
        
        return output

    def backward(self,input,gradLoss):
        
        upper_bound = self.nLayers -1
        lower_bound = -1
        decr = -1
        gradOutput = gradLoss.mm(self.Why.transpose(1,0))
        for t in range(upper_bound, lower_bound, decr):
            if t==0:
                h = self.h0
            else:
                h = self.layer[t-1].output
            gradOutput = self.layer[t].backward(gradOutput,input[:,t,:],h,self.Whh)
        self.gradBy = gradLoss.sum(dim = 0)
        self.gradWhy = self.hprev.transpose(1,0).mm(gradLoss)

    def update(self,learn_rate):
        
        cumlGradBh = torch.zeros_like(self.Bh).double()
        cumlGradWxh = torch.zeros_like(self.Wxh).double()
        cumlGradWhh = torch.zeros_like(self.Whh).double()

        for layer in self.layer:
            cumlGradBh += layer.gradBh.sum(dim=0)
            cumlGradWxh += layer.gradWxh
            cumlGradWhh += layer.gradWhh

            layer.gradWhh = 0
            layer.gradWxh = 0
            layer.gradBh = 0
            
        self.Wxh -= learn_rate*cumlGradWxh 
        self.Whh -= learn_rate*cumlGradWhh 
        self.Bh -= learn_rate*cumlGradBh
        self.Why -= learn_rate*self.gradWhy
        self.By -= learn_rate*self.gradBy
