from nltk import word_tokenize
import os
# from gensim.models import FastText  # FIXME: why does Sphinx dislike this import?
import fasttext

# Skipgram model
model = fasttext.skipgram('data.txt', 'model')
print(model.words) # list of words in dictionary

# CBOW model
model = fasttext.cbow('data.txt', 'model')