from nltk.stem.lancaster import LancasterStemmer
from nltk import word_tokenize
from numpy import array

__all__ = ['Parser']


class Parser:

    @staticmethod
    def __prepare(sentence: str) -> list:
        stemmer = LancasterStemmer()

        sentence_words = word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

        return sentence_words

    @staticmethod
    def convert(sentence: str, words: list) -> array:

        _sentence: list = Parser.__prepare(sentence)
        predictors: list = [0] * len(words)

        for s in _sentence:
            for i, w in enumerate(words):
                if w == s:
                    predictors[i] = 1

        return array(predictors).reshape(1, -1)
