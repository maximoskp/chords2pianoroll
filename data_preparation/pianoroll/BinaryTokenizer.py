import numpy as np

class BinaryTokenizer:
    def __init__(self, num_digits=12):
        self.vocab_size = 2**12;
    # end init
    
    def fit(self, corpus):
        pass;
    # end fit

    def transform(self, corpus):
        return corpus.dot(1 << np.arange(corpus.shape[-1] - 1, -1, -1))
    # end transform

    def fit_transform(self, corpus):
        return corpus.dot(1 << np.arange(corpus.shape[-1] - 1, -1, -1))
    # end transform
# end class BinaryTokenizer