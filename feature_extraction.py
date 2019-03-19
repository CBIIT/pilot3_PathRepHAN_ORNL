from gensim.models import Word2Vec
import numpy as np
import csv
import logging
import sys
import logging
import cPickle
import re
import random

#get filepath to pubmed abstracts
args = (sys.argv)
if len(args) != 2:
    raise Exception("Usage: python feature_extraction.py <path to pubmed abstracts csv>")
filename = args[1]

#set config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#class for reading csv
class Tsv(object):
   delimiter = ';'
   quotechar = '"'
   escapechar = '\\'
   doublequote = False
   skipinitialspace = True
   lineterminator = '\n'
   quoting = csv.QUOTE_ALL

#load data
class Abstracts(object):
    def __init__(self,filename):
        self.filename = filename
    def __iter__(self):
        with open(self.filename,'r') as f:
            reader = csv.reader(f,dialect=Tsv)
            for row in reader:
                yield row[1].split()

data = Abstracts(filename)

#train word2vec
model = Word2Vec(size=350,window=8,min_count=5,workers=4,alpha=0.025,min_alpha=0.025)
model.build_vocab(data)
for epoch in range(10):
    model.train(data,total_examples=model.corpus_count,epochs=1)
    model.alpha -= 0.002 #decrease the learning rate
    model.min_alpha = model.alpha #fix the learning rate, no decay
    model.save('data/word2vec.w2v')
    
#load word2vec
model = Word2Vec.load('data/word2vec.w2v')

#save all word embeddings to matrix
print "saving word vectors to matrix"
vocab = np.zeros((len(model.wv.vocab)+1,350))
word2id = {}

#first row of embedding matrix isn't used so that 0 can be masked
for key,val in model.wv.vocab.iteritems():
    idx = val.__dict__['index']
    idx += 1
    vocab[idx,:] = model[key]
    word2id[key] = idx
    
#normalize embeddings
vocab -= vocab.mean()
vocab /= (vocab.std()*2.5)

#add additional word embedding for unknown words
unk = np.expand_dims(np.mean(vocab,0),0)
vocab = np.concatenate((vocab, unk))
unk = len(vocab)-1

#save to disk
print 'saving vocab to disk'
np.save('data/vocab',vocab)
with open('data/word2id.pkl','w') as f:
    cPickle.dump(word2id,f,-1)
    
#process abstracts
labels2id = {}
max_sents = 0
max_words = 0
id = 0
print 'processing abstracts'
with open('data/han_input.csv','w') as w:
    writer = csv.writer(w,dialect=Tsv)
    with open(filename,'r') as r:
        reader = csv.reader(r,dialect=Tsv)
        i = 0
        for row in reader:
            i += 1
            
            #get labels
            label = row[0]
            
            #create label binarizer dictionary
            if label in labels2id:
                label = labels2id[label]
            else:
                labels2id[label] = id
                label = labels2id[label]
                id += 1
            
            #get sentences
            allsents = []
            sents = re.split(' \. | \? | \! ',row[1] + ' ')  #add space at end to split last punctuation mark
            for sent in sents:
                if len(sent) > 1:
                    
                    #convert words to ids
                    sent_ids = [str(word2id[word]) if word in word2id else str(unk) for word in sent.split()]
                
                    #get max words per sentence                    
                    if len(sent_ids) > max_words:
                        max_words = len(sent_ids)
                        
                    allsents.append(sent_ids)
                    
            allsents = [' '.join(ids) for ids in allsents]
            
            #make sure no empty docs
            if len(allsents) == 0:
                raise Exception('length zero document found')
                
            #get max sentences
            num_sents = len(allsents)
            if num_sents > max_sents:
                max_sents = num_sents
            
            #write to file
            writer.writerow([label,', '.join(allsents)])           
            sys.stdout.write("processed %i rows      \r" % i)
            sys.stdout.flush()
print

#save labels dic to disk
with open('data/labels2id.pkl','w') as f:
    cPickle.dump(labels2id,f,-1)

#shuffle records
fid = open("data/han_input.csv", "r")
li = fid.readlines()
fid.close()

random.shuffle(li)

fid = open("data/han_input.csv", "w")
fid.writelines(li)
fid.close()
