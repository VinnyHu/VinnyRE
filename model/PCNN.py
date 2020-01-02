"""
created by VinnyHu
Distant supervision for relation extraction via Piecewise  Convolutional Neural Networks
using mask to implement pici-wise pooling ,learned by openNRE
used for Relation Extraction
learn how to write great codes!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DefaultConfig
from .BasicModule import BasicModule

import math

class PCNN(BasicModule):

    def __init__(self,opt,token2id,vectors = None):
        super(PCNN,self).__init__()


        self.token2id = token2id
        self.max_length = opt.max_length
        self.num_token = len(token2id)  #the num of word

        self.num_pos = self.max_length * 2
        self.word_dim = opt.word_embedding_dim
        self.pos_dim = opt.pos_size
        self.kernel_num = opt.kernel_num
        self.input_size = self.word_dim + self.pos_dim * 2
        self.blank_padding = opt.blank_padding
        self.fc = nn.Linear(in_features=self.kernel_num * 3,out_features=opt.num_classes)

        # initial word2vec
        self.word_embedding = nn.Embedding(num_embeddings=self.num_token,embedding_dim=self.word_dim)
        if vectors is not None:
            print("Initializing word embedding with word2vec...")
            vectors = torch.from_numpy(vectors)
            if self.num_token == len(vectors) + 2:
                unk = torch.randn(1, self.word_dim) / math.sqrt(self.word_dim)
                blk = torch.zeros(1, self.word_dim)  # 是否可以改进，比如随机成一个分布
                self.word_embedding.weight.data.copy_(torch.cat([vectors, unk, blk], 0))
            else:
                self.word_embedding.weight.data.copy_(vectors)
            print('Finished!')

        #position embedding
        self.pos1_Embedding = nn.Embedding(num_embeddings=self.num_pos,embedding_dim=self.pos_dim,padding_idx=0)
        self.pos2_Embedding = nn.Embedding(num_embeddings=self.num_pos,embedding_dim=self.pos_dim,padding_idx=0)

        self.dropout = nn.Dropout(opt.dropout)

        self.padding_size = opt.padding_size
        self.act = F.relu
        self.softmax = nn.Softmax(-1)

        # convs = [
        #     nn.Sequential(
        #         nn.Conv1d(in_channels=self.input_size,
        #                   out_channels=self.kernel_num,
        #                   kernel_size=kernel_size,
        #                   padding=self.padding_size  #why?
        #                   )
        #     )
        #     for kernel_size in opt.kernel_sizes
        # ]
        # self.conv = nn.ModuleList(convs)
        self.conv =  nn.Conv1d(in_channels=self.input_size,
                          out_channels=self.kernel_num,
                          kernel_size=opt.kernel_sizes,
                          padding=self.padding_size  #why?
                          )
        self.pool = nn.MaxPool1d(self.max_length)

        self.tanh = F.tanh

        self.mask_embedding = nn.Embedding(4,3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]))
        self.mask_embedding.weight.requires_grad = False
        self.minus = -1000


    def forward(self,token,pos1,pos2,mask):

        # if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
        #     raise Exception("Size of token, pos1 ans pos2 should be (B, L)")

        x = torch.cat([self.word_embedding(token),
                       self.pos1_Embedding(pos1),
                       self.pos2_Embedding(pos2)
                       ],2) # B L EM

        x = x.transpose(1,2) # B EM L conv1d B in_channels L

        x = self.conv(x)    #B H L
        mask = 1 - self.mask_embedding(mask).transpose(1,2)

        pool1 = self.pool(self.act(x + self.minus * mask[:,0:1,:]))
        pool2 = self.pool(self.act(x + self.minus * mask[:,1:2,:]))
        pool3 = self.pool(self.act(x + self.minus * mask[:,2:3,:]))

        x = torch.cat([pool1,pool2,pool3],1)
        x = x.squeeze(2) # B 3H


        x = self.tanh(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x



