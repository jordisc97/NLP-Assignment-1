import numpy as np
import scipy as sp
import time
import sklearn.preprocessing

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm_notebook as tqdm


class TFIDFVectorizer():
    
    def __init__(self, vocabulary, word_to_ind, tokenize, normalize_tf=True, normalize_tfidf=True):
        self.tokenize = tokenize
        self.vocabulary = vocabulary
        self.word_to_ind = word_to_ind
        self.n_features = len(word_to_ind)
        self.normalize_tf = normalize_tf
        self.normalize_tfidf = normalize_tfidf
        self.X_w = None
        self.idf = None
        self.n_documents = None
    
    def fit(self, X):
        """
        Fit TFID vectorizer to a certain corpus of documents X
        """
        assert isinstance(X,list), "You should pass a list"
        
        t1 = time.time()
        self.__build_vocabulary(X)
        self.n_documents = len(X)
        self.__compute_idf()
        print('TFIDF fit finished in',str(round(time.time()-t1, 2)),'seconds')
        
    def transform(self, X):
        """
        Transform a corpus X to its TFID vectorization
        """
        assert self.X_w is not None and self.idf is not None and self.n_documents is not None,'Fit must be performed first'
        assert isinstance(X,list), "You should pass a list"
        
        t1 = time.time()
        col_indices = []
        row_indices = []
        sp_data     = []
        
        encoded_X = None # Create an encoded_X
        for m, doc in enumerate(X):
#             print(m)
            words = self.tokenize(doc)
            for w in words:
                if w in self.word_to_ind:
                    index = self.word_to_ind[w]
                    col_indices.append(index)
                    row_indices.append(m)
                    sp_data.append(1)
#             print(doc, normalize_tf)
#             print(len(doc))
#             tf = self.__term_frequency(doc, normalize_tf)
#             tfidf = tf.multiply(self.idf)
#             if normalize_tfidf: tfidf = tfidf/sp.sparse.linalg.norm(tfidf)
#             encoded_X = sp.vstack((encoded_X, tf)) if encoded_X is not None else tfidf
        encoded_X = sp.sparse.csr_matrix((sp_data, (row_indices, col_indices)), shape=(len(X), self.n_features))
        if self.normalize_tf: encoded_X = sklearn.preprocessing.normalize(encoded_X, axis=1)
        
        encoded_X = encoded_X.multiply(self.idf)
        if self.normalize_tfidf: encoded_X = sklearn.preprocessing.normalize(encoded_X, axis=1)
        
        print('TFIDF transform finished in',str(round(time.time()-t1, 2)),'seconds')
        return encoded_X
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __build_vocabulary(self, corpus):
        """
        This function builds X_w, a dict containing for each key, how
        many documents having that key are in our corpus.
        """
        X_w = {}

        for document in tqdm(corpus, desc="Building corpus: "):
            words = set(self.tokenize(document))
            for word in words:
                if word not in X_w: X_w[word] = 1
                else: X_w[word] += 1

        self.X_w = X_w
        
    def __compute_idf(self):
        col_indices = []
        row_indices = []
        sp_data     = []

        for w in self.X_w:
            docs_present = self.X_w[w]
            index = self.word_to_ind[w]
            col_indices.append(index)
            row_indices.append(0)
            sp_data.append( np.log(self.n_documents / (1 + docs_present)) )

        self.idf = sp.sparse.csr_matrix((sp_data, (row_indices, col_indices)), shape=(1, self.n_features))

    def __term_frequency(self, document, normalize=True):
        
        words = self.tokenize(document)
        col_indices = []
        row_indices = []
        sp_data     = []

        for w in words:
            if w in self.word_to_ind:
                index = self.word_to_ind[w]
                col_indices.append(index)
                row_indices.append(0)
                sp_data.append(1)
        
        tf = sp.sparse.csr_matrix((sp_data, (row_indices, col_indices)), shape=(1, self.n_features))
        
        if normalize:
            return tf.multiply(1/sp.sparse.linalg.norm(tf))
        else:
            return tf


