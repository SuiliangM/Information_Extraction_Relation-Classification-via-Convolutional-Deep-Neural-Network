from Data import Data
import glob
import fasttext
import re

data = Data('./2007', './2010')
# data.process_2007()
# data.split_2007()
# data.build_2007_corpus('corpus.txt')

# model = fasttext.skipgram('corpus.txt', 'model')
# model = fasttext.load_model('model.bin')
# print(model['I'])


train_files = glob.glob('split_2007/train_*.txt')
for file in train_files:
    num = re.search('train_(\d)', file).group(1)
    classifier = fasttext.supervised(file, 'models/{}'.format(num), epoch=50, lr=0.5, min_count=1)
    result = classifier.test('split_2007/dev_{}.txt'.format(num))

    precision = result.precision
    recall = result.recall
    f1 = 2 * precision * recall / (precision + recall)
    print(f1)