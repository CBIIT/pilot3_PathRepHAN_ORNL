## Hierarchical attention networks for information extraction from cancer pathology reports

##### Author: Biomedical Sciences, Engineering and Computing Group, Computer Sciences and Engineering Division, Oak Ridge National Laboratory

This repo contains some of the code used for the paper *Hierarchical attention networks for information extraction from cancer pathology reports*. 
BSEC group designed the model for information extraction from cancer pathology reports. Because the dataset used in our the paper is not open access, 
we instead provide an implementation that classifies Pubmed abstracts. 

A hierarchical attention network is a deep learning model composed of hierarchies of bidirectional LSTMs/GRUs with attention
mechanisms. The model has two "hierarchies". The lower hierarchy takes in one sentence at a time, broken into word embeddings. This hierarchy outputs a
weighted sentence embedding based on the words in the sentence that are most relevant to the classification. The upper hierarchy takes in one document at a
time, broken into the sentence embeddings from the lower hierarchy. This hierarchy outputs a weighted document embedding based on the sentences in the document that
are most relevant to the classification. Dropout is applied to this final document embedding, and it is then fed into a softmax classifier.

### Dataset Description

Each abstract in the PubMed dataset comes pretagged with medical subject headings (MeSH labels) that identify general topics associated with that abstract
(e.g. child development, cardiovascular disease). For our datasets, we only use MeSH labels that are explicitly listed as major topics for an
abstract. We extracted Pubmed abstracts associated with 8 selected labels -- metabolism, physiology, genetics, chemistry, pathology, surgery, psychology, 
and diagnosis -- and only use abstracts that are associated with a single label. We have curated a dataset with 1,000
documents of each label (8,000 abstracts).

### Installation

HAN is written and tested in `Python 3.6` with the following dependencies.

- TensorFlow: An open source machine learning framework
    - `pip install tensorflow`
- scikit-learn: Machine Learning in Python
    - `pip install scikit-learn`
- NumPy: The fundamental package for scientific computing with Python
    - `pip install numpy`
- Gensim: Python package for generating Word2Vec and other embeddings
    - `pip install gensim`

### Run with sample data

1. To prepare the data, execute the script `feature_extraction.py`.
```
$ python feature_extraction.py
```

2. To train a HAN model with the prepared data, execute the script `tf_han.py`.

```
$ python tf_han.py
loading data
processed 8000 rows
building hierarchical attention network
2019-04-04 12:33:19.130441: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2019-04-04 12:33:19.414967: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: GeForce GTX 970 major: 5 minor: 2 memoryClockRate(GHz): 1.329
pciBusID: 0000:01:00.0
totalMemory: 4.00GiB freeMemory: 3.30GiB
2019-04-04 12:33:19.421780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-04-04 12:33:19.793321: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-04-04 12:33:19.796625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-04-04 12:33:19.798192: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-04-04 12:33:19.800220: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3010 MB memory) -> physical GPU (device: 0, name: GeForce GTX 970, pci bus id: 0000:01:00.0, compute capability: 5.2)
training network on 7200 documents, validating on 800 documents
epoch 1, sample 7200 of 7200, loss: 0.065831
epoch 1 training accuracy: 68.2083%
epoch 1 validation accuracy: 70.6250%
```

### Disclaimer
UT-BATTELLE, LLC AND THE GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED. THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE. THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF, IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.


### Acknowledgments
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
