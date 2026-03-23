from sentence_transformers import SentenceTransformer
from core.base import BaseTextEncoder

class SentenceEncoder(BaseTextEncoder):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return texts
    
    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # TODO: add HF AutoTokenizer + AutoModel wrapper later