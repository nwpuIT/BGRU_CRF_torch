import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from utils.utils_tools import  *
class BGRU_CRF(nn.Module):
	def __init__(self,vocab_size,tag_to_ix,embedding_dim,hidden_dim):
		super(BGRU_CRF,self).__init__()
		self.embedding_dim=embedding_dim
		self.hidden_dim=hidden_dim
		self.vocab_size=vocab_size
		self.tag_to_ix=tag_to_ix# list to idx
		self.tagset_size=len(tag_to_ix)
		self.embedding=nn.Embedding(self.vocab_size,self.embedding_dim)
		self.rgu=nn.GRU(embedding_dim,hidden_dim,num_layers=1,bidirectional=True)
		self.hidde2tag=nn.Linear(hidden_dim,self.tagset_size)
		self.transition=nn.Parameter(torch.randn((self.tagset_size,self.tagset_size)))
		self.transition[self.tag_to_ix[START_TAG],:]=-10000
		self.transition[:,self.tag_to_ix[STOP_TAG]] = -10000

	def _get_GRU_features(self,sentence):
		#seq
		embed=self.embedding(sentence).view(len(sentence),1,-1)#seq,1,embed_dim
		outputs,hidden=self.gru(embed)
		outputs=outputs[:,:,:self.hidden_dim]+outputs[:,:,self.hidden_dim:]
		feats=self.hidde2tag(outputs.squeeze(1))  # seq,tagset
		return feats
	#已知路径的gold-score 得分
	def _score_sentence(self,feats,targets):
		score=torch.zeros(1)
		targets=torch.cat(torch.tensor([self.tag_to_ix[START_TAG]],dtype=torch.long),targets)
		for i ,feat in enumerate(feats):
			score+=self.transition[targets[i+1],targets[i]]+feat[targets[i+1]]
		score+=self.transition[self.tag_to_ix[STOP_TAG],targets[-1]]
		return score
	#score 我们的每一个Score都是对应于一个完整的路径，举例说【我 爱 中国人民】对应标签【N V N】那这个标签就是一个完整的路径，也就对应一个Score值
	#以下函数实现了所有词性标注的可能路径得分
	def _forward_alg(self, feats):
		init_alphas=torch.full((1,self.tagset_size),-1000)
		init_alphas[0][self.tag_to_ix[START_ATG]]=0;
		forward_var=init_alphas
		for feat in feats:#*target_size
			alphas_t=[]
			for next_tag in range(self.tagset_size):
			# broadcast the emission score: it is the same regardless of the previous tag
				emit_score=feat[next_tag].view(1,-1).expand(1,self.tagset_size)
				trans_score=self.transition[next_tag].view(1,-1)
				next_tag_var=forward_var+trans_score+emit_score#动态规划
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			forward_var=torch.cat(alphas_t).view(1,-1)#	到第（t - 1）step时５个标签的各自分数
		terminal_var=forward_var+self.transition[self.tag_to_ix[STOP_TAG]]
		return log_sum_exp(terminal_var)
	def neg_log_likelihood(self,sentence,target):
		feats=self._get_GRU_features(sentence)# seq,tagset
		forward_score=self._forward_alg(feats)
		gold_score=self._score_sentence(feats,target)
		return forward_score-gold_score
	#动态规划
	def _viterbi_decoder(self,lstm_feats):
		backpointers=[]
		init_vars=torch.full((1,self.tagset_size),-1000)
		init_vars[0][self.tag_to_ix[START_TAG]]=0
		# forward_var at step i holds the viterbi variables for step i-1
		forward_var=init_vars
		for feat in lstm_feats:
			bptrs_t=[]
			viterbivars_t=[]
			# next_tag_var[i] holds the viterbi variable for tag i at the
			# previous step, plus the score of transitioning
			# from tag i to next_tag.
			# We don't include the emission scores here because the max
			# does not depend on them (we add them in below)
			for next_tag in range(self.tagset_size):
				next_tag_var=forward_var+self.transition[next_tag]
				best_tag_id=torch.max(next_tag_var,1)[1].item()
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			# Now add in the emission scores, and assign forward_var to the set
			# of viterbi variables we just computed
			forward_var=(torch.cat(viterbivars_t)+feat).view(1,-1)
			backpointers.append(bptrs_t)
		terminal_var=forward_var+self.transition[self.tag_to_ix[STOP_TAG]]
		best_tag_id=torch.max(terminal_var,1)[1].item()
		path_score=terminal_var[0][best_tag_id]
		best_path=[best_tag_id]
		# Follow the back pointers to decode the best path.
		for bptrs_t in reversed(backpointers):
			best_tag_id=bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		start=best_path.pop()
		assert  start==self.tag_to_ix[START_TAG]
		best_path.reverse()
		return path_score,best_path
	def forward(self,sentence):
		lstm_feats=self._get_GRU_features(sentence)# seq,tagset
		score,tag_seq=self._viterbi_decoder(lstm_feats)
		return  score,tag_seq




