def data_load():
    import json
    from Dataset import Dataset

    with open('Base/data.json') as json_file:
        d = dict(json.load(json_file))

    return d, Dataset(d, ['?', '!'])


def model_load():
    from Model import Model
    from os.path import exists

    x_train, y_train = dataset.load_data()

    model = Model()
    exists_model = exists('Base/model.h5')

    if exists_model:
        model.load_model('Base/model.h5')
    else:
        model.build_nn(x_train.shape[1], y_train.shape[1])

        model.fit(x_train, y_train, 1000, 5)
        model.save('Base/model.h5')

    return model


def talk(context, update):
    request = Parser.convert(update.message.text, dataset.words)
    resp = chatBot.response(request)

    if resp >= 0:
        words = data[dataset.target[resp]]['response']
        words = words[randint(0, len(words))]

        context.send_message(chat_id=update.effective_chat.id, text=words)
    else:
        context.send_message(chat_id=update.effective_chat.id, text="Não consegui entender ':(")


if __name__ == '__main__':
    from Parser import Parser
    from numpy.random import randint
    from telegram.ext import Filters, MessageHandler, Updater

    data, dataset = data_load()
    chatBot = model_load()

    telegram = Updater(token="SEU_TOKEN_AQUI")
    telegram.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), talk))
    telegram.start_polling()
    telegram.idle()
