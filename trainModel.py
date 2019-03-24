import model
import criterion
from model import *
from criterion import *
import torch 
import torch.utils.data
from random import sample
import numpy as np
import itertools
##################################################
print("Loading data and labels\n")

#if(True):
# data I/O
data_raw = open('train_data.txt', 'r').readlines() # should be simple plain text file
data_raw_test = open('test_data.txt', 'r').readlines() # should be simple plain text file
data_test = [j.split() for j in data_raw_test]


''' getting number of sentences'''
data = [i.split() for i in data_raw]

unique = list(set(list(itertools.chain.from_iterable(data_test)) + list(itertools.chain.from_iterable(data))))

label_raw = open('train_labels.txt', 'r').read().split()
l = [len(i) for i in data]

data_size, vocab_size = len(data), len(unique)
labels = [int(label) for label in label_raw]
''' in case of one hot representation
number of words in vocab will decide the dimension of the one hot vector '''

char_to_ix = { ch:i for i,ch in enumerate(unique) }
ix_to_char = { i:ch for i,ch in enumerate(unique) }
d=list(char_to_ix.keys())

def onehot(number):
	
    oh = np.zeros(len(unique))
    index = char_to_ix[number]
    oh[index]=1
    
    return oh

def in_to_onehot(input):
	l=[]
	for i in input:
	    l.append(onehot(i))
	return l
	
zero_vector = np.zeros(len(unique))

#app = lambda input_vector,zero_extension : input_vector.append((zero_extended*zero_extension)[0])
def app(input_vector,zero_extension):
	for i in range(zero_extension):
	    input_vector.append(zero_vector)

no_layers = 20
hidden_dim = 6 
learn_rate = 0.01
epoch = 80
batch_size = 16
#vocab_size = 149

#model_0 = model(batch_size,hidden_dim,vocab_size,vocab_size,True)
#criterion = criterion()
input = [in_to_onehot(d) for d in data]
loss = 0
temp = [app(i,2720-len(i)) for i in input]

in_test = [in_to_onehot(d_test) for d_test in data_test]
data = torch.tensor(input)

####################################################
#else:

#	label_raw = open('train_labels.txt', 'r').read().split()
#	labels = [int(label) for label in label_raw]

#	no_layers = 20
#	hidden_dim = 5
#	learn_rate = 0.01
#	epoch = 60
#	batch_size = 16
#	vocab_size = 149
#	data = torch.load('train_tensor.txt').double()



hprev = torch.zeros(batch_size,hidden_dim).double()
model_0 = model(no_layers,hidden_dim,batch_size,vocab_size,True)
criterion = criterion()

min_batch = torch.arange(0,data.shape[0],batch_size)

print("\n.......Training Started.......\n")
for e in range(epoch):

    index = sample(range(0, data.shape[0]), data.shape[0])
    input_vect = data[index]
    label_int = [labels[i] for i in index]
    avg_epoch_loss = 0
    for batch in min_batch:
        output,hprev = model_0.forward(input_vect[batch:batch+batch_size,:,:],hprev)
        loss = criterion.forward(output,label_int[batch:batch+batch_size])
        gradLoss = criterion.backward(output,label_int[batch:batch+batch_size])
        avg_epoch_loss += loss.sum()/batch_size
        model_0.backward(input_vect[batch:batch+batch_size,:,:],gradLoss)
        ''' loss for one batch'''
        model_0.update(learn_rate)
    print("Epoch:"+str(e)+" avg_Training_loss: "+str(avg_epoch_loss/len(min_batch)))


torch.save(model_0,'model2.bin')

print("\nTesting started ..............\n")

file = open('test_max.csv','w')
for i in range(len(data_raw_test)):
    
    input_vect = torch.tensor(in_test[i])
    hprev = torch.zeros(hidden_dim).double()

    output = model_0.forward_test(input_vect,hprev)
    print(output.shape)
    out = list(output.numpy()[0])
    predict = { val:inx for inx,val in enumerate(out) }
    predicted_label = predict[max(out)]
    file.write(str(predicted_label)+"\n")
    
file.close()


















