def data_load():
    import json
    from Dataset import Dataset

    with open('Base/data.json') as json_file:
        d = dict(json.load(json_file))

    return d, Dataset(d, ['?'])


def model_load():
    from Agent import Agent
    from os.path import exists

    x_train, y_train = dataset.load_data()

    model = Agent()
    exists_model = exists('Model/model.h5')

    if exists_model:
        model.load_model('Model/model.h5')
    else:
        model.build_nn(x_train.shape[1], y_train.shape[1])

        model.fit(x_train, y_train, 1000, 5, 'Model/weights.h5')
        model.save('Model/model.h5')

    return model


def talk(text):
    request = Parser.convert(text, dataset.words)
    resp = chatBot.response(request)

    if resp >= 0:
        words = data[dataset.target[resp]]['response']
        words = words[randint(0, len(words))]

        print(words)
    else:
        print("Não consegui entender ':(")

    return resp


if __name__ == '__main__':
    from Parser import Parser
    from numpy.random import randint

    data, dataset = data_load()
    chatBot = model_load()

    while True:
        inp = input("")
        response = talk(inp)

        if response == 1:
            break
