from abc import ABC, abstractmethod


class BaseTextEncoder(ABC):
    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def encode(self, texts):
        pass