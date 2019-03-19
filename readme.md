## Hierarchical attention networks for information extraction from cancer pathology reports

This repo contains some of the code used for the paper *Hierarchical attention networks for information extraction from cancer pathology reports* (J Am Med Inform Assoc. 2017 Nov 16. doi: 10.1093/jamia/ocx131). 
Because the dataset used our the paper is proprietary, we instead provide an implementation that classifies Pubmed abstracts. 

### Dataset Description

Each abstract in the PubMed dataset comes pretagged with medical subject headings (MeSH labels) that identify general topics associated with that abstract
(e.g. child development, cardiovascular disease). For our datasets, we only use MeSH labels that are explicitly listed as major topics for an
abstract. We extracted Pubmed abstracts associated with 8 selected labels -- metabolism, physiology, genetics, chemistry, pathology, surgery, psychology, 
and diagnosis -- and only use abstracts that are associated with a single label. We have curated three datasets of difference sizes, the first with 1,000
documents of each label (8,000 abstracts), the second with 10,000 documents of each label (80,000 abstracts), and the third with 100,000 documents of each 
label (800,000 abstracts).

We also provide a simple feature extractor that uses gensim's word2vec to create 350-dimension word embeddings for each dataset. 

### Hierarchical Attention Networks

A hierarchical attention network is a deep learning model composed of hierarchies of bidirectional LSTMs/GRUs with attention
mechanisms. The model has two "hierarchies". The lower hierarchy takes in one sentence at a time, broken into word embeddings. This hierarchy outputs a
weighted sentence embedding based on the words in the sentence that are most relevant to the classification. The upper hierarchy takes in one document at a
time, broken into the sentence embeddings from the lower hierarchy. This hierarchy outputs a weighted document embedding based on the sentences in the document that
are most relevant to the classification. Dropout is applied to this final document embedding, and it is then fed into a softmax classifier.

### Instructions to Run Models

Assumes that you have a functional version of Python 2.7 preferably with Anaconda installation and a NVIDIA GPU-based backend. The requirements to run the software are in the requirements.txt file attached with this. 

Run the following to calculate the test set accuracy (0.9/0.1 train/test split) on the selected dataset:

Traditional machine learning models:
 - python traditional_ml.py \<path to PubMed abstracts csv\>
 - python xg-boost.py \<path to PubMed abstracts csv\>

Deep Learning Models: 
 - python feature_extraction.py \<path to PubMed abstracts csv\>
 - python tf_rnn.py
 - python th_cnn.py
 - python tf_han.py
 
Note: \<path to PubMed abstracts csv\> can be 'data/labeled_abstracts_reduced_8000.csv', 'data/labeled_abstracts_reduced_80000.csv', or 'data/labeled_abstracts_reduced_800000.csv'

### Acknowledgements
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
