import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
# returns a python float  #var是Variable,维度是１
def to_scalar(var):
	return var.view(-1).data.tolist()[0]

def prepare_sequence(seq,to_ix):
	idxs=[to_ix[word] for word in  seq]
	return (torch.LongTensor(idxs))

def log_sum_exp(vec):
	max_score=vec[0,torch.max(vec,1)[1].item()]
	max_score_broadcast=max_score.view(1,-1).expand(1,vec.size(1))#1*5
	#第一步就是exp操作，这个很容易让计算机上溢。因此先让所有的值减去最大值，最后再加回来就好了，公式推导回去还是原来公式
	return  max_score+torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))
