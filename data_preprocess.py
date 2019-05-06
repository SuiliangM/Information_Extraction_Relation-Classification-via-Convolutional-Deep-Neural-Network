import os
from nltk import word_tokenize
from nltk.corpus import wordnet
from tqdm import tqdm
import re

class Data(object):
    def __init__(self, train_folder):
        self.train_folder = train_folder

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
            
            sentence = sentence.replace("<e1>", " E1_START ").replace("</e1>", " E1_END ")
            sentence = sentence.replace("<e2>", " E2_START ").replace("</e2>", " E2_END ")

            tokens = word_tokenize(sentence)        
            tokens = self._clean_tokens(sentence_num, tokens)
            
            fOut.write(" ".join([ label, " ".join(tokens) ]))
            fOut.write("\n")
        fOut.close()
            
        print(outputpath, "created")

    def preprocess2007(self, filepath, outputpath):    
        fOut = open(outputpath, 'w')
        lines = [line.strip() for line in open(filepath)]
        for line in lines:
            if 'Comment' in line:
                lines.remove(line)
        
        for idx in range(0, len(lines), 3):
            sentence_num = lines[idx][:3]
            sentence = lines[idx][4:-1]
            labels = lines[idx+1]
            
            sentence = sentence.replace("<e1>", " E1_START ").replace("</e1>", " E1_END ")
            sentence = sentence.replace("<e2>", " E2_START ").replace("</e2>", " E2_END ")

            tokens = word_tokenize(sentence)        
            tokens = self._clean_tokens(sentence_num, tokens)
            
            label = re.search('"(true|false)"', labels).group(1)
            
            out_line = "__lablle__{}, {}\n".format(label, " ".join(tokens[1:]))
            fOut.write(out_line)
        fOut.close()
            
        print(outputpath, "created")

data = Data('./2010')
# data.preprocess2010('./2010/TRAIN_FILE.TXT', 'output.txt')
data.preprocess2007('2007/relation-1-train.txt', '2007.txt')