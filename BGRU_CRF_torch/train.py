from  utils.utils_tools import  *
from  BGRU_CRF import BGRU_CRF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word2idx={}
for sentent ,tag, in training_data:
	for word in sentent:
		if word not in word2idx.keys():
			word2idx[word]=len(word2idx)
tag2idx={"B":0,"I":1,"O":2,START_TAG:3,STOP_TAG:4}

model=BGRU_CRF(len(word2idx),tag2idx,5,4)
optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
	precheck_sent=prepare_sequence(training_data[0][0],word2idx)
	precheck_tags=torch.tensor([tag2idx[t] for t in training_data[0][1]],dtype=torch.long)
	print(model(precheck_sent))

for epoch in range(300):
	for sentent ,tag in training_data:
		optimizer.zero_grad()
		sentent_id=prepare_sequence(sentent,word2idx)
		targets=torch.tensor([tag2idx[t] for t in tag],dtype=torch.long)
		loss=model.neg_log_likelihood(sentent_id,targets)
		loss.backward()
		optimizer.step()

# Check predictions after training
with torch.no_grad():
	precheck_sent=prepare_sequence(training_data[0][0],word2idx)
	print(model(precheck_sent))
