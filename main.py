# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CNN import CNN
from Dataset import SemEvalDataset
from utils import load_data, load_embedding, get_args, accuracy, F1


def train():
    args = get_args()

    # Load data
    dataset = SemEvalDataset(args.train_filename, max_len=args.seq_len)
    dataloader = DataLoader(dataset, args.batch_size, True,
                            num_workers=args.num_workers)
    dataset_val = SemEvalDataset(
        args.test_filename, max_len=args.seq_len, d=(dataset.d, dataset.rel_d))
    dataloader_val = DataLoader(
        dataset_val, args.batch_size, True, num_workers=args.num_workers)
    
    args.word_embedding = load_embedding(args.embedding_filename, args.embedding_wordlist_filename,
                                         dataset.d)
    args.vac_len_word = len(dataset.d.word2id)
    args.vac_len_rel = len(dataset.rel_d.word2id)
    args.dw = args.word_embedding.shape[1]

    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))

    # Build models
    writer = SummaryWriter()
    model = CNN(args)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_eval_acc = 0.
    best_eval_f1 = 0.

    for i in range(args.nepoch):
        # Training
        total_loss = 0.
        total_acc = 0.
        total_f1 = 0.

        ntrain_batch = 0
        model.train()
        for (seq, e1, e2, dist1, dist2, r) in dataloader:
            ntrain_batch += 1
            seq = Variable(seq)
            e1 = Variable(e1)
            e2 = Variable(e2)
            dist1 = Variable(dist1)
            dist2 = Variable(dist2)
            r = Variable(r)
            r = r.view(r.size(0))

            pred = model(seq, dist1, dist2, e1, e2)
            l = loss_func(pred, r)
            acc = accuracy(pred, r)
            f1 = F1(pred, r)
            total_acc += acc
            total_f1 += f1
            total_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        writer.add_scalar('train/loss', l.item(), i)
        writer.add_scalar('train/accuracy', total_acc / ntrain_batch, i)
        writer.add_scalar('train/f1', total_f1 / ntrain_batch, i)

        print("Epoch: {}, Training loss : {:.4}, acc: {:.4}, f1: {:.4}".
              format(i, total_loss/ntrain_batch, total_acc/ntrain_batch, total_f1/ntrain_batch))

        # Evaluation
        if i % args.eval_every == args.eval_every - 1:
            val_total_acc = 0.
            val_total_f1 = 0.

            nval_batch = 0
            model.eval()
            for (seq, e1, e2, dist1, dist2, r) in dataloader_val:
                nval_batch += 1
                seq = Variable(seq)
                e1 = Variable(e1)
                e2 = Variable(e2)
                dist1 = Variable(dist1)
                dist2 = Variable(dist2)
                r = Variable(r)
                r = r.view(r.size(0))

                pred = model(seq, dist1, dist2, e1, e2)
                acc = accuracy(pred, r)
                f1 = F1(pred, r)
                val_total_acc += acc
                val_total_f1 += f1
            best_eval_acc = max(best_eval_acc, val_total_acc/nval_batch)

            writer.add_scalar('test/accuracy', val_total_acc/nval_batch, i)
            writer.add_scalar('test/F1', val_total_f1/nval_batch, i)
            print("Epoch: {}, Val acc: {:.4f}, F1: {:.4f}".
                  format(i, val_total_acc/nval_batch, val_total_f1/nval_batch))
    print('Best acc: {}'.format(best_eval_acc))
    print('Best F1: {}'.format(best_eval_f1))
    torch.save(model.state_dict(), args.model_file)
    writer.close()


if __name__ == '__main__':
    train()
