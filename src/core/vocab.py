from collections import Counter

class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.stoi = {'<pad>': 0, '<unk>': 1}
        self.itos = ['<pad>', '<unk>']

    def fit(self, tokenized_texts: list[list[str]]) -> None:
        counter = Counter()

        for tokens in tokenized_texts:
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.stoi)
                self.itos.append(token)
    
    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]
    
    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[idx] for idx in ids]