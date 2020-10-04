from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from numpy.random import shuffle
from numpy import array

__all__ = ['Dataset']


class Dataset:

    def __init__(self, data: dict, ignore: list) -> None:
        self.__data: dict = data
        self.__ignore = ignore

        self.__doc: list = []
        self.__words:  list = []
        self.__target: list = []
        self.__dataset: tuple = ()

    @property
    def words(self) -> list:
        return self.__words

    @property
    def target(self) -> list:
        return self.__target

    def load_data(self) -> tuple:
        self.__separate(self.__ignore)
        return self.__dataset

    def __separate(self, ignore: list) -> None:
        for key in self.__data.keys():
            for r in self.__data[key]['request']:
                w = word_tokenize(r)

                self.__words.extend(w)
                self.__doc.append((w, key))

            self.__target.append(key)

        stemmer = LancasterStemmer()
        self.__words = sorted(list(set([stemmer.stem(w.lower()) for w in self.__words if w not in ignore])))

        self.__dataset = self.__genDataset()

    def __genDataset(self) -> tuple:
        stemmer: LancasterStemmer = LancasterStemmer()
        dataset: list = []

        for doc in self.__doc:
            predictors: list = []

            request = doc[0]
            request = [stemmer.stem(w.lower()) for w in request]

            for w in self.__words:
                predictors.append(1) if w in request else predictors.append(0)

            target: list = [0] * len(self.__target)
            target[self.__target.index(doc[1])] = 1

            dataset.append([predictors, target])

        shuffle(dataset)

        return self.__convertToArrays(dataset)

    def __convertToArrays(self, dataset: list) -> tuple:

        x_train: list = []
        y_train: list = []

        for predictors, target in dataset:
            x_train.append(predictors)
            y_train.append(target)

        return array(x_train), array(y_train)
