import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
import csv

#get filepath to pubmed abstracts
args = (sys.argv)
if len(args) != 2:
    raise Exception("Usage: python feature_extraction.py <path to pubmed abstracts csv>")
filename = args[1]

class Tsv(object):
   delimiter = ';'
   quotechar = '"'
   escapechar = '\\'
   doublequote = False
   skipinitialspace = True
   lineterminator = '\n'
   quoting = csv.QUOTE_ALL

#load corpus
docs = []
labels = []
with open(filename,'r') as r:
    reader = csv.reader(r,dialect=Tsv)
    for i,row in enumerate(reader):         
        docs.append(row[1])
        labels.append(row[0])
        sys.stdout.write("processed %i rows      \r" % i)
        sys.stdout.flush()
print

#tfidf vectorization
print 'vectorizing documents'
vectorizer = TfidfVectorizer(min_df=10, stop_words='english',ngram_range=(1, 2))
X = vectorizer.fit_transform(docs)

#label encoder
le = LabelEncoder()
y = le.fit_transform(labels)

#kfold cross validation
splits = 10
kf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=1234)

#classify - replace clf with desired classifier and settings
print "training classifier"
scores = []
i = 0
for train_index, test_index in kf.split(X,y):
    i += 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

    print "Classifier - kfold %i of %i accuracy: %.4f%%" % (i,splits,score*100)
    
print "Classifier - overall accuracy: %.4f" % (np.mean(scores)*100)
