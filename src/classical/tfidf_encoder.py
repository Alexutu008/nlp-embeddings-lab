from sklearn.feature_extraction.text import TfidfVectorizer
from core.base import BaseTextEncoder

class TfidfEncoder(BaseTextEncoder):
    def __init__(self, max_features=5000, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(
            max_features= max_features,
            ngram_range= ngram_range,
            lowercase= True
        )
    
    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self
    
    def encode(self, texts):
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out().tolist()


# TODO: implement TF-IDF from scratch later