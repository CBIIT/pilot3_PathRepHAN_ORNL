import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import sys

class hierarchical_attention_network(object):
    '''
    hierarchical attention network for document classification
    https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num classes: int
        number of output classes
      - max_sents: int
        maximum number of sentences per document
      - max_words: int
        maximum number of words per sentence
      - rnn_type: string (default: "gru")
        rnn cells to use, can be "gru" or "lstm"
      - rnn_units: int (default: 200)
        number of rnn units to use for embedding layers
      - attention_size: int (default: 300)
        number of dimensions to use for attention context layer
      - dropout_keep: float (default: 0.5)
        dropout keep rate for final softmax layer
       
    methods:
      - train(data,labels,epochs=30,savebest=False,filepath=None)
        train network on given data
      - predict(data)
        return the one-hot-encoded predicted labels for given data
      - score(data,labels,bootstrap=False,bs_samples=100)
        return the accuracy of predicted labels on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
    '''
    
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,rnn_type="gru",
                 rnn_units=200,attention_size=300,dropout_keep=0.5):

        self.rnn_units = rnn_units
        if rnn_type == "gru":
            self.rnn_cell = GRUCell
        elif rnn_type == "lstm":
            self.rnn_cell = LSTMCell
        else:
            raise Exception("rnn_type parameter must be set to gru or lstm")
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words

        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_sents,max_words])
        words_per_line = tf.reduce_sum(tf.sign(self.doc_input),1)
        num_lines = tf.reduce_sum(tf.sign(words_per_line))
        max_words_ = tf.reduce_max(words_per_line)
        doc_input_reduced = self.doc_input[:num_lines,:max_words_]
        num_words = words_per_line[:num_lines]

        #word rnn layer
        word_embeds = tf.gather(tf.get_variable('embeddings',initializer=
                      embedding_matrix.astype(np.float32),dtype=tf.float32),doc_input_reduced)
        with tf.variable_scope('words'):
            [word_outputs_fw,word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    word_embeds,sequence_length=num_words,dtype=tf.float32)
        word_outputs = tf.concat((word_outputs_fw, word_outputs_bw),2)
        
        #word attention
        seq_mask = tf.reshape(tf.sequence_mask(num_words,max_words_),[-1])
        word_u = tf.layers.dense(tf.reshape(word_outputs,[-1,self.rnn_units*2]),attention_size,tf.nn.tanh,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        word_exps = tf.layers.dense(word_u,1,tf.exp,False,kernel_initializer=tf.contrib.layers.xavier_initializer())
        word_exps = tf.where(seq_mask,word_exps,tf.ones_like(word_exps)*0.000000001)
        word_alpha = tf.reshape(word_exps,[-1,max_words_,1])
        word_alpha /= tf.reshape(tf.reduce_sum(word_alpha,1),[-1,1,1])
        sent_embeds = tf.reduce_sum(word_outputs*word_alpha,1)
        sent_embeds = tf.expand_dims(sent_embeds,0)

        #sentence rnn layer
        with tf.variable_scope('sentence'):
            [sent_outputs_fw,sent_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    sent_embeds,sequence_length=tf.expand_dims(num_lines,0),dtype=tf.float32)
        sent_outputs = tf.concat((tf.squeeze(sent_outputs_fw,[0]),tf.squeeze(sent_outputs_bw,[0])),1)
        
        #sentence attention
        sent_u = tf.layers.dense(sent_outputs,attention_size,tf.nn.tanh,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        sent_exp = tf.layers.dense(sent_u,1,tf.exp,False,kernel_initializer=tf.contrib.layers.xavier_initializer())
        sent_atten = sent_exp/tf.reduce_sum(sent_exp)
        doc_embed = tf.transpose(tf.matmul(tf.transpose(sent_outputs),sent_atten))

        #classification functions
        logits = tf.layers.dense(doc_embed,num_classes,kernel_initializer=tf.orthogonal_initializer())
        self.prediction = tf.nn.softmax(logits)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=self.labels_rs)
        self.optimizer = tf.train.AdamOptimizer(0.0001,0.9,0.99).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
    
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros((self.ms,self.mw))
            for i,line in enumerate(inputval):
                for j, word in enumerate(line):
                    retval[i,j] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")
    
    def train(self,data,labels,epochs=30,validation_data=None,savebest=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - epochs: int (default: 30)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - savebest: boolean (default: False)
            set to True to save the best model based on validation score per epoch
          - filepath: string (optional)
            path to save model if savebest is set to True
        
        outputs:
            None
        '''
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")
        
        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)
        
        print('training network on %i documents, validating on %i documents' \
              % (len(data), validation_size))
        
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
            print()
            trainscore = correct/len(data)
            print("epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100))
            
            #validate
            if validation_data:
                score = self.score(validation_data[0],validation_data[1])
                print("epoch %i validation accuracy: %.4f%%" % (i+1, score*100))
                
            #save if performance better than previous best
            if savebest and score >= prevbest:
                prevbest = score
                self.save(filepath)

    def predict(self,data):
        '''
        return the one-hot-encoded predicted labels for given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
        
        outputs:
            numpy array of one-hot-encoded predicted labels for input data
        '''
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
        
    def score(self,data,labels,bootstrap=False,bs_samples=100):
        '''
        return the accuracy of predicted labels on given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
        
        outputs:
            float representing accuracy of predicted labels on given data
        '''        
        correct = 0.
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct +=1

        accuracy = correct/len(labels)
        return accuracy
        
    def save(self,filename):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load model weights from a file
        
        parameters:
          - filepath: string
            path from which to load model weights
        
        outputs:
            None
        '''
        self.saver.restore(self.sess,filename)
        
if __name__ == "__main__":

    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.model_selection import train_test_split
    import pickle
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
    max_sents = 0
    max_words = 0
    docs = []
    labels = []
       
    #load saved files
    print("loading data")
    vocab = np.load('data/vocab.npy')
    with open('data/word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)
       
    #load corpus
    with open('data/han_input.csv','r') as r:
        reader = csv.reader(r,dialect=Tsv)
        for i,row in enumerate(reader):
            sents = [sent.split() for sent in row[1].split(', ')]
            if len(sents) > max_sents:
                max_sents = len(sents)
            l = len(max(sents,key=len))
            if l > max_words:
                max_words = l
                
            docs.append(sents)
            labels.append(row[0])
            
            sys.stdout.write("processed %i rows      \r" % (i+1))
            sys.stdout.flush()
    print()
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
    print("building hierarchical attention network")
    nn = hierarchical_attention_network(vocab,classes,max_sents,max_words)
    nn.train(X_train,y_train,epochs=2,validation_data=(X_test,y_test))
