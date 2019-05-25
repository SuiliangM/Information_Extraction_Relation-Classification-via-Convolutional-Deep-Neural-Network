from collections import Counter

import numpy as np
import torch

from torch.utils.data import Dataset
from utils import load_data

class SemEvalDataset(Dataset):
    '''
    Dataset object
    '''
    def __init__(self, filename, max_len, d=None):
        '''
        
        '''
        seqs, e1_pos, e2_pos, rs = load_data(filename)
        self.max_len = max_len
        if d is None:
            self.d = build_dict(seqs)
            self.rel_d = build_dict([[r] for r in rs], add_extra=False)
        else:
            self.d = d[0]
            self.rel_d = d[1]
        self.seqs, self.e1s, self.e2s, self.dist1s, self.dist2s =\
            self.vectorize_seq(seqs, e1_pos, e2_pos)
        self.rs = np.array([[self.rel_d.word2id[r]] for r in rs])

    def vectorize_seq(self, seqs, e1_pos, e2_pos):
        '''
        '''
        new_seqs = np.zeros((len(seqs), self.max_len))
        dist1s = np.zeros((len(seqs), self.max_len))
        dist2s = np.zeros((len(seqs), self.max_len))
        e1s = np.zeros((len(seqs), 1))
        e2s = np.zeros((len(seqs), 1))
        for r, (seq, e1_p, e2_p) in enumerate(zip(seqs, e1_pos, e2_pos)):
            seq = list(
                map(lambda x: self.d.word2id[x] if x in self.d.word2id else 0, seq))
            dist1 = list(map(map_pos, [idx - e1_p[1]
                                       for idx, _ in enumerate(seq)]))  # Last word
            dist2 = list(map(map_pos, [idx - e2_p[1]
                                       for idx, _ in enumerate(seq)]))
            e1s[r] = seq[e1_p[1]]
            e2s[r] = seq[e2_p[1]]
            for i in range(min(self.max_len, len(seq))):
                new_seqs[r, i] = seq[i]
                dist1s[r, i] = dist1[i]
                dist2s[r, i] = dist2[i]
        return new_seqs, e1s, e2s, dist1s, dist2s

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = torch.from_numpy(self.seqs[index]).long()
        e1 = torch.from_numpy(self.e1s[index]).long()
        e2 = torch.from_numpy(self.e2s[index]).long()
        dist1 = torch.from_numpy(self.dist1s[index]).long()
        dist2 = torch.from_numpy(self.dist2s[index]).long()
        r = torch.from_numpy(self.rs[index]).long()
        return seq, e1, e2, dist1, dist2, r


class Dictionary(object):
    def __init__(self):
        self.word2id = {}
        self.id2word = []

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)


def build_dict(seqs, add_extra=True, dict_size=100000):
    d = Dictionary()
    cnt = Counter()
    for seq in seqs:
        cnt.update(seq)
    d = Dictionary()
    if add_extra:
        d.add_word(None)  # 0 for not in the dictionary
    for word, cnt in cnt.most_common()[:dict_size]:
        d.add_word(word)
    return d


def map_pos(p):
    if p < -60:
        return 0
    elif -60 <= p < 60:
        return p + 61
    else:
        return 121
