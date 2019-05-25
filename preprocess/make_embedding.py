lines = open('hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt').readlines()
out_embedding = open('./data/embeddings.txt', 'w')
out_wordlist = open('./data/words.lst', 'w')
for line in lines:
    word = line.split()[0]
    embed = line.split()[1:]
    out_wordlist.write(word + '\n')
    out_embedding.write(" ".join(embed) + '\n')