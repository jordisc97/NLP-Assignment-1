import scipy
import scipy.sparse as sp
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from collections import defaultdict
import sklearn
import re
stemmer =  SnowballStemmer(language='english')


class SimpleCountVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self,
                 min_word_counts=1,
                 doc_cleaner_pattern=r"[^a-zA-Z]",
                 token_pattern=r"(?u)\b\w\w+\b",
                 dtype=np.float32,
                 doc_cleaner_func=None,
                 tokenizer_func=None,
                 word_transformer_func=None):
        
        self._retype = type(re.compile('hello, world'))

        self.min_word_counts     = min_word_counts
        self.doc_cleaner_pattern = doc_cleaner_pattern
        self.token_pattern       = token_pattern
        self.dtype               = dtype
        
        self.doc_cleaner_func      = doc_cleaner_func
        self.tokenizer_func        = tokenizer_func
        self.word_transformer_func = word_transformer_func

        self.vocabulary = set()
        self.word_to_ind = {}


    def build_doc_cleaner(self, lower=True):
        """
        Returns a function that cleans undesirable substrings in a string.
        It also lowers the input string if lower=True
        """
        if self.doc_cleaner_func:
            return self.doc_cleaner_func
        else:
            if isinstance(self.doc_cleaner_pattern, self._retype):
                #clean_doc_pattern = self.doc_cleaner_pattern.sub(" ", doc)
                clean_doc_pattern = re.compile(self.doc_cleaner_pattern)
            else:
                clean_doc_pattern = re.compile(self.doc_cleaner_pattern)

            if lower:
                 return lambda doc: clean_doc_pattern.sub(" ", doc).lower()
            else:
                 return lambda doc: clean_doc_pattern.sub(" ", doc)

    def build_tokenizer(self):
        """Returns a function that splits a string into a sequence of tokens"""
        if self.tokenizer_func:
            return self.tokenizer_func
        
        else:
            token_pattern = re.compile(self.token_pattern)
            return lambda doc: token_pattern.findall(doc)

    def build_word_transformer(self):
        """Returns a stemmer or lemmaitzer if object has any"""
        
        if self.word_transformer_func:
            return self.word_transformer_func
        else:
            return lambda word: word
        
    def tokenize(self, doc):
        doc_cleaner      = self.build_doc_cleaner()
        doc_tokenizer    = self.build_tokenizer()
        doc     = doc_cleaner(doc)
        words = doc_tokenizer(doc)
            
        return words
        
    def fit(self, X):

        assert self.vocabulary == set(), "self.vocabulary is not empty it has {} words".format(len(self.vocabulary))
        assert isinstance(X,list), "X is expected to be a list of documents"
        
        i = 0
        word_to_ind = {}
        doc_cleaner      = self.build_doc_cleaner()
        doc_tokenizer    = self.build_tokenizer()
        word_transformer = self.build_word_transformer()
        
        for x in X:
            words = doc_cleaner(x)
            words = doc_tokenizer(words)
            for word in words:
                if word not in word_to_ind:
                    word_to_ind[word] = i
                    i += 1
        self.word_to_ind = word_to_ind
        self.n_features = len(word_to_ind)    
                
        self.vocabulary = set(word_to_ind.keys())
                
        return self
    
    def transform(self, X, memory_efficient=False):
        
        doc_cleaner      = self.build_doc_cleaner()
        doc_tokenizer    = self.build_tokenizer()
        word_transformer = self.build_word_transformer()      
        
        col_indices = []
        row_indices = []
        sp_data     = []
                
        if memory_efficient:
            for m, x in enumerate(X):  
                words = doc_cleaner(x)
                words = doc_tokenizer(words)
                for word in words: 
                    index = self.word_to_ind[word]
                    
                    col_indices.append(index)
                    row_indices.append(m)
                    sp_data.append(1)
                    
            encoded_X = sp.csr_matrix((sp_data, (row_indices, col_indices)), shape = (len(X) ,self.n_features)) 
                        
        else:
            
            encoded_X = np.zeros((len(X), len(self.word_to_ind)))
            for m, x in enumerate(X):  
                words = doc_cleaner(x)
                words = doc_tokenizer(words)
                for word in words: 
                    index = self.word_to_ind[word]
                    encoded_X[m, index] += 1
        
        return encoded_X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        encoded_X = self.transform(X)
        return encoded_X
    
    def _words_in_vocab(self, X):
        
        if isinstance(X, str):
            return [w for w in self.tokenize(X) if w in self.vocabulary]
        
        X_words_in_vocab = []
        for sentence in X:
            X_words_in_vocab.append(self.tokenize(sentence))
            
        return X_words_in_vocab