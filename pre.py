from nltk import word_tokenize
import os

trainFile = 'Train/TRAIN_FILE.TXT'
def clean_tokens(sent_num, tokens):
    ret = []
    for t in tokens:
        t = t.strip().split()
        if len(t) > 1: 
            print(sent_num, t)
        t = "_".join(t)
        ret.append(t)
    return ret 

def preprocess2010(filepath, outputpath):    
    fOut = open(outputpath, 'w')
    lines = [line.strip() for line in open(filepath)]
    for idx in range(0, len(lines), 4):
        sentence_num = lines[idx].split("\t")[0]
        sentence = lines[idx].split("\t")[1][1:-1]
        label = lines[idx+1]
        
        sentence = sentence.replace("<e1>", " E1_START ").replace("</e1>", " E1_END ")
        sentence = sentence.replace("<e2>", " E2_START ").replace("</e2>", " E2_END ")
        #sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        #sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")

        tokens = word_tokenize(sentence)        
        tokens = clean_tokens(sentence_num, tokens)
        
        fOut.write(" ".join([ label, " ".join(tokens) ]))
        fOut.write("\n")
    fOut.close()
        
    print(outputpath, "created")

def preprocess2007(filepath, outputpath):    
    fOut = open(outputpath, 'w')
    lines = [line.strip() for line in open(filepath)]
    for idx in range(0, len(lines), 4):
        sentence_num = lines[idx][:3]
        sentence = lines[idx][4:-1]
        label = lines[idx+1]
        
        sentence = sentence.replace("<e1>", " E1_START ").replace("</e1>", " E1_END ")
        sentence = sentence.replace("<e2>", " E2_START ").replace("</e2>", " E2_END ")
        #sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        #sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")

        tokens = word_tokenize(sentence)        
        tokens = clean_tokens(sentence_num, tokens)
        
        fOut.write(" ".join([ label, " ".join(tokens) ]))
        fOut.write("\n")
    fOut.close()
        
    print(outputpath, "created")

# preprocess2010(trainFile, 'out.txt')
preprocess2007('train.txt', 'output.txt')