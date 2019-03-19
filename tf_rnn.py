import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sys
import operator
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class text_rnn(object):

    def __init__(self,embedding_matrix,num_classes,max_words,rnn_type="gru",
                 rnn_units=200,attention_context=300,dropout_keep=0.5):

        self.rnn_units = rnn_units
        if rnn_type == "gru":
            self.rnn_cell = GRUCell
        elif rnn_type == "lstm":
            self.rnn_cell = LSTMCell
        else:
            raise Exception("rnn_type parameter must be set to gru or lstm")
        self.dropout_keep = dropout_keep
        self.vocab = embedding_matrix
        self.embedding_size = embedding_matrix.shape[1]
        self.max_words = max_words

        #shared variables
        with tf.variable_scope('rnn'):
            self.word_W = tf.Variable(self._ortho_weight(2*rnn_units,attention_context),name='word_W')
            self.word_b = tf.Variable(np.asarray(np.zeros(attention_context),dtype=np.float32),name='word_b')
            self.word_context = tf.Variable(self._ortho_weight(attention_context,1),name='word_context')
        with tf.variable_scope('classify'):
            self.W_softmax = tf.Variable(self._ortho_weight(rnn_units*2,num_classes),name='W_softmax')
            self.b_softmax = tf.Variable(np.asarray(np.zeros(num_classes),dtype=np.float32),name='b_softmax')
        
        #word embeddings
        with tf.variable_scope('embeddings'):
            self.embeddings = tf.cast(tf.Variable(embedding_matrix,name='embeddings'),tf.float32)
        self.dropout = tf.placeholder(tf.float32)
        
        #sentence input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_words])
        self.word_mask = tf.not_equal(self.doc_input,tf.zeros_like(self.doc_input))
        self.word_nonzero = tf.boolean_mask(self.doc_input,self.word_mask)
        self.word_embeds = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings,self.word_nonzero),0)
        self.doc_len = self._length(self.word_embeds)
        with tf.variable_scope('rnn'):
            [self.word_outputs_fw,self.word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    self.word_embeds,sequence_length=self.doc_len,dtype=tf.float32)
        self.word_outputs = tf.concat((tf.squeeze(self.word_outputs_fw,[0]),tf.squeeze(self.word_outputs_bw,[0])),1)
        self.word_atten = tf.squeeze(tf.map_fn(self._attention_step,self.word_outputs),[1,2])
        self.word_atten = self.word_atten/tf.reduce_sum(self.word_atten)
        self.doc_embed = tf.matmul(tf.expand_dims(self.word_atten,0),self.word_outputs)
        self.doc_embed_drop = tf.nn.dropout(self.doc_embed,self.dropout)

        #classification functions
        self.output = tf.matmul(self.doc_embed_drop,self.W_softmax)+self.b_softmax
        self.prediction = tf.nn.softmax(self.output)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.999).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
    
    def _length(self,sequence):
        '''
        return length of dynamic input tensor for rnn
        '''
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
       
    def _attention_step(self,embedding):
        '''
        get attention multiplier across words
        '''
        embedding = tf.expand_dims(embedding,0)
        u = tf.nn.tanh(tf.matmul(embedding,self.word_W) + self.word_b)
        return tf.exp(tf.matmul(u,self.word_context))
        
    def _ortho_weight(self,fan_in,fan_out):
        '''
        generate orthogonal weight matrix
        '''
        bound = np.sqrt(2./(fan_in+fan_out))
        W = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u
        else:
            W = v
        return W.astype(np.float32)
    
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros(self.max_words)
            for i,word in enumerate(inputval):
                retval[i] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")
    
    def train(self,data,labels,epochs=30,validation_data=None,savebest=False,filepath=None):

        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")
        
        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)
        
        print 'training network on %i documents, validating on %i documents' \
              % (len(data), validation_size)
        
        #track best model for saving
        prevbest = 0    
        for i in range(epochs):
            correct = 0.
            
            #train
            for doc in range(len(data)):
                inputval = self._list_to_numpy(data[doc])
                feed_dict = {self.doc_input:inputval,self.labels:labels[doc],self.dropout:self.dropout_keep}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],feed_dict=feed_dict)
                if np.argmax(pred) == np.argmax(labels[doc]):
                    correct += 1
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (i+1,doc+1,len(data),cost))
                sys.stdout.flush()
            print ""
            trainscore = correct/len(data)
            print "epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100)
            
            #validate
            if validation_data:
                score = self.score(validation_data[0],validation_data[1])
                print "epoch %i validation accuracy: %.4f%%" % (i+1, score*100)
                
            #save if performance better than previous best
            if savebest and score >= prevbest:
                prevbest = score
                self.save(filepath)

    def predict(self,data):

        labels = []
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            prob = np.squeeze(prob,0)
            one_hot = np.zeros_like(prob)
            one_hot[np.argmax(prob)] = 1
            labels.append(one_hot)
        
        labels = np.array(labels)
        return labels
        
    def score(self,data,labels):

        #count correct predictions
        correct = []
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct.append(1.)
            else:
                correct.append(0.)
        
        accuracy = np.sum(correct)/len(labels)
        return accuracy
        
    def save(self,filename):

        self.saver.save(self.sess,filename)

    def load(self,filename):

        self.saver.restore(self.sess,filename)
        
if __name__ == "__main__":

    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.model_selection import train_test_split
    import cPickle
    import os
    import csv
    
    class Tsv(object):
       delimiter = ';'
       quotechar = '"'
       escapechar = '\\'
       doublequote = False
       skipinitialspace = True
       lineterminator = '\n'
       quoting = csv.QUOTE_ALL
       
    #corpus info
    maxlen = 0  
    docs = []
    labels = []
       
    #load saved files
    print "loading data"
    vocab = np.load('data/vocab.npy')
    with open('data/word2id.pkl', 'r') as f:
        word2id = cPickle.load(f)
       
    #load corpus
    with open('data/han_input.csv','r') as r:
        reader = csv.reader(r,dialect=Tsv)
        for i,row in enumerate(reader):
            words = [idx.replace(",", "") for idx in row[1].split()]
            if len(words) > maxlen:
                maxlen = len(words)
            docs.append(words)
            labels.append(row[0])
            
            sys.stdout.write("processed %i rows      \r" % i)
            sys.stdout.flush()
    print
    num_docs = i+1

    #label encoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = len(le.classes_)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    #test train split
    X_train,X_test,y_train,y_test = train_test_split(docs,y_bin,test_size=0.1,
                                    random_state=1234,stratify=y)
    
    #train nn
    print "building text rnn"
    nn = text_rnn(vocab,classes,maxlen)
    
    if not os.path.exists('./savedmodels'):
        os.makedirs('./savedmodels')
    nn.train(X_train,y_train,epochs=5,validation_data=(X_test,y_test),
             savebest=True,filepath='./savedmodels/rnn.ckpt')
    
    #load best nn
    nn.load('./savedmodels/rnn.ckpt')
    acc = nn.score(X_test,y_test)
    y_pred = np.argmax(nn.predict(X_test),1)
    print "RNN - test set accuracy: %.4f" % (acc*100)
