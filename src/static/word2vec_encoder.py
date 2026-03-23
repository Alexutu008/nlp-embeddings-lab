import numpy as np
from gensim.models import Word2Vec
from core.base import BaseTextEncoder

def simple_tokenize(text):
    return text.lower().split()

class Word2VecEncoder(BaseTextEncoder):
    def __init__(self, vector_size=100, window=5, min_count=1, sg=1, workers=1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg # 0: CBOW  1: Skip-Gram
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def fit(self, texts):
        tokenized = [simple_tokenize(text) for text in texts]
        self.model = Word2Vec(
            sentences = tokenized,
            vector_size = self.vector_size,
            window = self.window,
            min_count = self.min_count,
            sg = self.sg,
            workers = self.workers,
            epochs = self.epochs
        )        
        return self
    
    def encode(self, texts):
        vectors = []
        for text in texts:
            tokens = simple_tokenize(text)
            token_vecs = [self.model.wv[token] for token in tokens if token in self.model.wv]

            if len(token_vecs) == 0:
                vectors.append(np.zeros(self.vector_size))
            else:
                vectors.append(np.mean(token_vecs, axis=0)) #(num_tokens, vector_size) => (vector_size,)
        return np.vstack(vectors)
    
    def most_similar(self, word, topn=5):
        return self.model.wv.most_similar(word, topn=topn)
    
    # TODO: implement Skip-Gram and CBOW from scratch later