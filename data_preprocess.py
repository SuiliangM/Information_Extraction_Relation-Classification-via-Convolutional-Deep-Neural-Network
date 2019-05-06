import glob
import os
import re

import fasttext
from nltk import word_tokenize
from nltk.corpus import wordnet
from tqdm import tqdm
import fasttext

class Data(object):
    def __init__(self, folder_2007, file_2010):
        self.folder_2007 = folder_2007
        self.file_2010 = file_2010

    def _clean_tokens(self, sent_num, tokens):
        ret = []
        for t in tokens:
            t = t.strip().split()
            if len(t) > 1:
                print(sent_num, t)
            t = "_".join(t)
            ret.append(t)
        return ret

    def preprocess2010(self, filepath, outputpath):
        fOut = open(outputpath, 'w')
        lines = [line.strip() for line in open(filepath)]
        for idx in range(0, len(lines), 4):
            sentence_num = lines[idx].split("\t")[0]
            sentence = lines[idx].split("\t")[1][1:-1]
            label = lines[idx+1]

            sentence = sentence.replace(
                "<e1>", " E1_START ").replace("</e1>", " E1_END ")
            sentence = sentence.replace(
                "<e2>", " E2_START ").replace("</e2>", " E2_END ")

            tokens = word_tokenize(sentence)
            tokens = self._clean_tokens(sentence_num, tokens)

            fOut.write(" ".join([label, " ".join(tokens)]))
            fOut.write("\n")
        fOut.close()

        print(outputpath, "created")

    def process_2007(self):
        files = glob.glob('2007/*.txt')
        for file in files:
            num = re.search('2007\/relation-(\d)-train.txt', file).group(1)
            file_name = 'processed_2007/train_{}.txt'.format(num)
            self._process_2007_file(file, file_name)

    def _process_2007_file(self, filepath, outputpath):
        fOut = open(outputpath, 'w')
        lines = [line.strip() for line in open(filepath)]
        for line in lines:
            if re.match('^Comment:', line):
                lines.remove(line)

        for idx in range(0, len(lines), 3):
            sentence_num = lines[idx][:3]
            sentence = lines[idx][4:-1]
            labels = lines[idx+1]

            sentence = sentence.replace(
                "<e1>", " E1_START ").replace("</e1>", " E1_END ")
            sentence = sentence.replace(
                "<e2>", " E2_START ").replace("</e2>", " E2_END ")

            tokens = word_tokenize(sentence)
            tokens = self._clean_tokens(sentence_num, tokens)

            label = re.search('"(true|false)"', labels).group(1)

            out_line = "__label__{}, {}\n".format(label, " ".join(tokens[1:]))
            fOut.write(out_line)
        fOut.close()

        print(outputpath, "created")

    def split_2007(self):
        files = glob.glob('processed_2007/*.txt')
        for file in files:
            num = re.search('train_(\d)', file).group(1)
            lines = open(file).readlines()
            assert len(lines) == 140

            train_out = open('split_2007/train_{}.txt'.format(num), 'w')
            dev_out = open('split_2007/dev_{}.txt'.format(num), 'w')
            train_part = lines[:100]
            dev_part = lines[100:]
            train_out.write("".join(train_part))
            dev_out.write("".join(dev_part))

    def build_2007_corpus(self, output_path):
        files = glob.glob('processed_2007/*.txt')
        output = open(output_path, 'w')
        for file in files:
            lines = open(file).readlines()
            for line in lines:
                output_line = " ".join(line.split()[1:]) + '\n'
                output.write(output_line)


data = Data('./2007', './2010')
# data.process_2007()
data.split_2007()
# data.build_2007_corpus('corpus.txt')

# model = fasttext.skipgram('corpus.txt', 'model')
# model = fasttext.load_model('model.bin')
# print(model['I'])


train_files = glob.glob('split_2007/train_*.txt')
for file in train_files:
    num = re.search('train_(\d)', file).group(1)
    classifier = fasttext.supervised(file, 'models/{}'.format(num), epoch=25, lr=1.0, min_count=1)
    result = classifier.test('split_2007/dev_{}.txt'.format(num))

    precision = result.precision
    recall = result.recall
    f1 = 2 * precision * recall / (precision + recall)
    print(f1)