from numpy import array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


__all__ = ['Model']


class Model:

    def __init__(self):
        self.__model: Sequential = None

    def build_nn(self, inp_shape: int, out_shape: int) -> None:
        model = Sequential()

        model.add(Dense(units=16, activation='sigmoid', kernel_initializer='random_uniform', input_dim=inp_shape))
        model.add(BatchNormalization())
        model.add(Dense(units=16, activation='sigmoid', kernel_initializer='random_uniform'))
        model.add(BatchNormalization())
        model.add(Dense(units=out_shape, activation='sigmoid', kernel_initializer='random_uniform'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        self.__model = model

    def load_model(self, path: str) -> None:
        self.__model = load_model(path)

    def fit(self, x: array, y: array, epochs: int, batch_size: int) -> None:

        if self.__model is not None:
            EarStop = EarlyStopping(monitor='loss', min_delta=1e-10, patience=20, verbose=1)
            RedPlat = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=1)

            self.__model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[EarStop, RedPlat])
        else:
            raise Exception("O modelo não foi criado ou carregado.")

    def response(self, x: array) -> int:
        index: int = -1

        if self.__model is not None:
            predict: array = self.__model.predict(x).round(0).astype('int')

            if True in (predict == 1):
                index = predict.argmax()
        else:
            raise Exception("O modelo não foi criado ou carregado.")

        return index

    def save(self, path: str):
        self.__model.save(path)
