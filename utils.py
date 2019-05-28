# coding=utf-8
import argparse

import numpy as np
import torch

def load_data(filename):
    '''
    Load the SemEval data set
    '''
    seqs = []
    e1_pos = []
    e2_pos = []
    rs = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            data[0] = data[0].lower().split(' ')
            seqs.append(data[0])
            e1_pos.append((int(data[1]), int(data[2])))
            e2_pos.append((int(data[3]), int(data[4])))
            rs.append(data[5])
    return seqs, e1_pos, e2_pos, rs


def load_embedding(embedding_file, word_list_file, d):
    word_list = {}
    with open(word_list_file) as f:
        for i, line in enumerate(f):
            word_list[line.strip()] = i

    with open(embedding_file, 'r') as f:
        lines = f.readlines()

    def process_line(line):
        return list(map(float, line.split(' ')))
    lines = list(map(process_line, lines))

    val_len = len(d.id2word)
    dw = len(lines[0])
    embedding = np.random.uniform(-0.01, 0.01, size=[val_len, dw])
    num_pretrained = 0
    for i in range(1, val_len):
        if d.id2word[i] in word_list:
            embedding[i, :] = lines[word_list[d.id2word[i]]]
            num_pretrained += 1

    print('#pretrained: {}, #vocabulary: {}'.format(num_pretrained, val_len))
    return embedding


def get_args():
    # Default parameters
    parser = argparse.ArgumentParser("CNN")
    parser.add_argument("--dim_posidx", type=int, default=5)
    parser.add_argument("--dim_conv", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--vac_len_rel", type=int, default=19)
    parser.add_argument("--vac_len_pos", type=int, default=122)
    parser.add_argument("--nepoch", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=str, default="3,4,5")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_filename', default='data/2010/merge_bin_2010/train/train-1')
    parser.add_argument('--test_filename', default='data/2010/merge_bin_2010/test/test-1')
    parser.add_argument('--model_file', default='./cnn.pt')
    parser.add_argument('--embedding_filename',
                        default='./embeddings/100/embeddings.txt')
    parser.add_argument('--embedding_wordlist_filename',
                        default='./embeddings/100/words.lst')

    args = parser.parse_args()
    args.kernel_sizes = list(map(int, args.kernel_sizes.split(',')))

    return args


def accuracy(preds, labels):
    n = preds.size(0)
    preds = torch.max(preds, dim=1)[1]
    correct = (preds == labels).sum()
    acc = correct.cpu().item() / n
    return acc

def F1(preds, labels):
    n = preds.size(0)
    preds = torch.max(preds, dim=1)[1]

    correct = (preds == labels).sum()
    prediction_true = preds.sum()
    all_true = labels.sum()

    precision = correct / prediction_true
    recall = correct / all_true
    f1 = 2 * precision * recall / (precision + recall)
    return f1