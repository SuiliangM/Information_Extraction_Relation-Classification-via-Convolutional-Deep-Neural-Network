# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class CNN(nn.Module):
    '''
    CNN model
    '''
    def __init__(self, args):
        super(CNN, self).__init__()

        self.word_embedding = nn.Embedding(args.vac_len_word, args.dw)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(args.word_embedding))
        self.pos_embedding_pos1 = nn.Embedding(
            args.vac_len_pos, args.dim_posidx)
        self.pos_embedding_pos2 = nn.Embedding(
            args.vac_len_pos, args.dim_posidx)

        self.dropout = nn.Dropout(args.dropout_rate)

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(args.dw + 2 * args.dim_posidx, args.dim_conv,
                      kernel_size=kernel_size, padding=int((kernel_size - 0) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(args.seq_len)
        ) for kernel_size in args.kernel_sizes])

        self.fc = nn.Linear(args.dim_conv * len(args.kernel_sizes) +
                            args.dw * 2, args.vac_len_rel)

    def forward(self, W, W_pos1, W_pos2, e1, e2):
        '''
        Network forwarding function
        '''
        # Load embeddings for words and position index features
        e1 = self.word_embedding(e1)
        e2 = self.word_embedding(e2)
        W = self.word_embedding(W)
        W_pos1 = self.pos_embedding_pos1(W_pos1)
        W_pos2 = self.pos_embedding_pos2(W_pos2)

        Wa = torch.cat([W, W_pos1, W_pos2], dim=2)

        conv = [conv(Wa.permute(0, 2, 1)) for conv in self.convs]
        conv = torch.cat(conv, dim=1)

        conv = self.dropout(conv)

        e_concat = torch.cat([e1, e2], dim=1)

        all_concat = torch.cat(
            [e_concat.view(e_concat.size(0), -1), conv.view(conv.size(0), -1)], dim=1)

        out = self.fc(all_concat)

        out = F.softmax(out)

        return out
