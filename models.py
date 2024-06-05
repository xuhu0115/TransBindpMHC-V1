import math
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(199901)
from scipy import interp
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from collections import OrderedDict
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import difflib

#plt.rc('font',family='Times New Roman')
#plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

seed = 199901
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

pep_max_len = 15 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len

vocab = np.load('./Dataset/vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)


# Transformer Parameters
d_model = 128                       # 这里可以修改的更大
d_emb = 64         # Embedding Size
d_ff = 512
d_k = d_v = 64  # dimension of K(=Q), V       

batch_size = 1024

#n_layers, n_heads, fold = 5, 1, 4

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

def read_predict_data(predict_data, batch_size =1024):
    print('# Samples = ', len(predict_data))
    
    if 'HLA_sequence' not in predict_data.columns and 'HLA' in predict_data.columns: 
        hla_sequence = pd.read_csv('./Dataset/HLA/new_hla_pseudo.csv')
        predict_data = pd.merge(predict_data, hla_sequence, on = 'HLA')
    elif 'HLA' not in predict_data.columns and 'HLA_sequence' in predict_data.columns: 
        hla_sequence = pd.read_csv('./Dataset/HLA/new_hla_pseudo.csv')
        predict_data = pd.merge(predict_data, hla_sequence, on = 'HLA_sequence')
              
    pep_inputs, hla_inputs = make_data(predict_data)
    data_loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs), batch_size, shuffle = False, num_workers = 0)
    return predict_data, pep_inputs, hla_inputs, data_loader

def make_data(data):
    pep_inputs, hla_inputs = [], []
    if 'peptide' not in data.columns: 
        peptides = data.mutation_peptide
    else:
        peptides = data.peptide
        
    for pep, hla in zip(peptides, data.HLA_sequence):
        pep, hla = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
        pep_input = [[vocab[n] for n in pep]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs

    def __len__(self): # 样本数
        return self.pep_inputs.shape[0] # 改成hla_inputs也可以哦！

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx]
    


class PositionalEncoding(nn.Module):
    """位置编码------为了使⽤序列的顺序信息，我们通过在输⼊表⽰中添加 
    位置编码（positional encoding）来注⼊绝对的或相对的位置信息。
    假设输⼊表⽰X ∈ Rn×d 包含⼀个序列中n个词元的d维嵌⼊表⽰。位置编码使⽤相同形状的位置嵌⼊矩阵P ∈ Rn×d 输出X + P """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model=d_model
        pe = torch.zeros(max_len, self.d_model)   # 创建⼀个⾜够⻓的P
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))  #math.log返回 x 的自然对数（底为 e ）
        pe[:, 0::2] = torch.sin(position * div_term)  #偶数
        pe[:, 1::2] = torch.cos(position * div_term)  #奇数
        ## 上面代码获取之后得到的pe:[max_len*d_emb]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_emb]
        pe = pe.unsqueeze(0).transpose(0, 1)   #返回一个新的张量，对输入的既定位置插入维度 1  ；调换数组的行列值的索引值
        self.register_buffer('pe', pe)   # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)





def get_attn_pad_mask(seq_q, seq_k):
    ## 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
    ## len_input * len*input  代表每个单词对其余包含自己的单词的影响力

    ## 所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算计算softmax之前会把这里置为无穷大；

    ## 一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要

    ## seq_q 和 seq_k 不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]




class ScaledDotProductAttention(nn.Module):
    """缩放点积注意⼒"""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn


# In[17]:


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;

    def __init__(self,d_model,n_layers, n_heads, n_fold):
        super(MultiHeadAttention, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_fold = n_fold
        self.d_model=d_model
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(self.d_model, d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * d_v, self.d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(device)(output + residual), attn


# In[18]:


class PoswiseFeedForwardNet(nn.Module):
    """基于位置的前馈⽹络"""
    #对序列中的所有位置的表⽰进⾏变换时使⽤的是同⼀个多层感知机（MLP），这就是称前馈⽹络是基于位置的（positionwise）的原因

    def __init__(self,d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model=d_model
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, self.d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(device)(output + residual) # [batch_size, seq_len, d_model]    #残差连接+层规范化


# In[19]:


class EncoderLayer(nn.Module):
    """EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络"""

    def __init__(self,n_layers, n_heads, n_fold):
        super(EncoderLayer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_fold = n_fold
        self.enc_self_attn = MultiHeadAttention(d_emb,self.n_layers, self.n_heads, self.n_fold)
        self.pos_ffn = PoswiseFeedForwardNet(d_emb)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


# In[20]:

# Encoder-1
class Encoder(nn.Module):
    """Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络"""

    def __init__(self,n_layers, n_heads, n_fold):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_fold = n_fold
        self.src_emb = nn.Embedding(vocab_size, d_emb)  # 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_emb)  # 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer(self.n_layers, self.n_heads, self.n_fold) for _ in range(1)])  # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # 下面这个代码通过src_emb，进行索引定位
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_emb]   #把数字索引转换成对应的向量
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_emb]  # 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]  #get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# ### Decoder

# In[21]:


class DecoderLayer(nn.Module):
    def __init__(self,n_layers, n_heads, n_fold):
        super(DecoderLayer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_fold = n_fold
        self.dec_self_attn = MultiHeadAttention(d_model,self.n_layers, self.n_heads, self.n_fold)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, dec_inputs, dec_self_attn_mask): # dec_inputs = enc_outputs
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


# In[22]:

#Encoder-2
class Decoder(nn.Module):
    def __init__(self,n_layers, n_heads, n_fold):
        super(Decoder, self).__init__()
#       self.tgt_emb = nn.Embedding(d_model * 2, d_model)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_fold = n_fold
        self.use_cuda = use_cuda
        device = torch.device("cuda" if self.use_cuda else "cpu")
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(self.n_layers, self.n_heads, self.n_fold) for _ in range(self.n_layers)])
        self.tgt_len = tgt_len
        
    def forward(self, dec_inputs): # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
#         dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device) # [batch_size, tgt_len, d_model]
#         dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
            
        return dec_outputs, dec_self_attns


# ### Transformer

# In[23]:


class Transformer(nn.Module):
    """分为三个部分：编码层，解码层，输出层"""
    def __init__(self,n_layers, n_heads, n_fold):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_fold = n_fold
        self.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")

        self.pep_encoder = Encoder(self.n_layers, self.n_heads, self.n_fold).to(device)
        self.hla_encoder = Encoder(self.n_layers, self.n_heads, self.n_fold).to(device)
        self.l1=nn.Linear(d_emb,d_model).to(device)     #新加这里
        self.decoder = Decoder(self.n_layers, self.n_heads, self.n_fold).to(device)
        self.tgt_len = tgt_len
        self.projection = nn.Sequential(
                                        nn.Linear(tgt_len * d_model, 256),
                                        nn.ReLU(True),

                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 64),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(64, 2)   #修改了这里
                                        ).to(device)
        
    def forward(self, pep_inputs, hla_inputs):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        
        enc_output = torch.cat((pep_enc_outputs, hla_enc_outputs), 1) # concat pep & hla embedding
        #print(enc_output.shape)  #torch.Size([1024, 49, 128])
        enc_outputs=self.l1(enc_output)   #新添加这里

        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1) # Flatten [batch_size, tgt_len * d_model]
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, hla_enc_self_attns, dec_self_attns

   

    
def eval_step(model, val_loader, threshold = 0.5, use_cuda = False):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
    torch.manual_seed(199901)
    torch.cuda.manual_seed(199901)
    with torch.no_grad():
        y_prob_val_list, dec_attns_val_list = [], []
        for val_pep_inputs, val_hla_inputs in val_loader:
            val_pep_inputs, val_hla_inputs = val_pep_inputs.to(device), val_hla_inputs.to(device)
            val_outputs, _, _, val_dec_self_attns = model(val_pep_inputs, val_hla_inputs)

            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_prob_val_list.extend(y_prob_val)
            
            dec_attns_val_list.extend(val_dec_self_attns[0][:, :, 15:, :15]) # 只要（34,15）行HLA，列peptide
                    
        y_pred_val_list = transfer(y_prob_val_list, threshold)
    
    return y_pred_val_list, y_prob_val_list, dec_attns_val_list


def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])




