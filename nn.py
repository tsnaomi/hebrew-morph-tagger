'''
This file is where I developed the neural network for the morphological tagger.
'''

import numpy as np
import os.path as op

from collections import defaultdict

from keras.layers import (
    Bidirectional,
    concatenate,
    Dense,
    Embedding,
    LSTM,
    TimeDistributed,
    )
from keras.models import Input, load_model, Model
from tabulate import tabulate

from data import Data


# filenames -------------------------------------------------------------------

MODELS_DIR = op.join(op.dirname(__file__), 'models')
MODEL_FN = op.join(MODELS_DIR, 'morph_tagger')


# data ------------------------------------------------------------------------

data = Data()  # load data!


# parameters ------------------------------------------------------------------

# for now, I am hard-coding arbitrary values for different parameters required
# by the network
WORD_EMBEDDING_DIM = 150
CHAR_EMBEDDING_DIM = 200
CHAR_LSTM_DIM = 100
LSTM_DIM = 100
RECURRENT_DROPOUT = 0.5
DROPOUT = 0.4
BATCH_SIZE = 32
EPOCHS = 10


# base experiment model -------------------------------------------------------

class BaseTagger:

    def __init__(self, model_fn, train_X, train_Y, output_n, activation, loss):
        if not model_fn.endswith('.h5'):
            model_fn += '.h5'

        self.model_fn = model_fn
        self.output_n = output_n
        self.activation = activation
        self.loss = loss

        try:
            self.model = load_model(self.model_fn)
            self.model.summary()

        except (OSError, TypeError, ValueError):
            self.def_architecture()
            self.train(train_X, train_Y)

    def def_architecture(self):
        # input/embeddings for words
        word_input = Input(shape=(data.max_sent_len, ))
        word_embedding = Embedding(
            input_dim=data.n_words,
            output_dim=WORD_EMBEDDING_DIM,
            input_length=data.max_sent_len,
            mask_zero=True,
        )(word_input)

        # input/embeddings for characters
        char_input = Input(shape=(data.max_sent_len, data.max_word_len, ))
        char_embedding = TimeDistributed(
            Embedding(
                input_dim=data.n_chars,
                output_dim=CHAR_EMBEDDING_DIM,
                input_length=data.max_word_len,
                mask_zero=True,
            ))(char_input)

        char_lstm = TimeDistributed(
            Bidirectional(
                LSTM(
                    units=CHAR_LSTM_DIM,
                    return_sequences=False,
                    recurrent_dropout=RECURRENT_DROPOUT,
                )))(char_embedding)

        # full embedding
        full_embedding = concatenate([word_embedding, char_lstm])

        # main LSTM
        lstm = Bidirectional(
            LSTM(
                units=LSTM_DIM,
                return_sequences=True,
                recurrent_dropout=RECURRENT_DROPOUT,
            ))(full_embedding)

        # output
        output = TimeDistributed(
            Dense(
                self.output_n,
                activation=self.activation,
            ))(lstm)

        self.model = Model([word_input, char_input], output)

        self.model.compile(
            optimizer='adam',
            loss=self.loss,
            metrics=['acc', ]
            )

        self.model.summary()

    def train(self, train_X, train_Y):
        self.model.fit(
            train_X,
            train_Y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            )

        try:
            # save the trained model to file!
            self.model.save(self.model_fn)

        except (OSError, TypeError, ValueError):
            pass  # welp

    def predict(self, test_X):
        return self.model.predict(
            test_X,
            batch_size=BATCH_SIZE,
            verbose=1,
            )


# morphological tagger --------------------------------------------------------

class MorphTagger(BaseTagger):

    def __init__(self, model_fn=MODEL_FN, train_X=None, train_Y=None):
        super().__init__(
            model_fn=model_fn,
            train_X=train_X,
            train_Y=train_Y,
            output_n=data.output_len,
            activation='sigmoid',
            loss='binary_crossentropy',
            )

    @staticmethod
    def acc(predictions, gold, threshold=0.5, force_pos=True):
        task_acc = defaultdict(float)
        acc = 0.
        n = 0.

        for p_sent, g_sent in zip(predictions, gold):
            for p_word, g_word in zip(p_sent, g_sent):

                if 1 in g_word:  # ignore sentence padding vectors
                    p_word = data.binarize(
                        p_word,
                        threshold=threshold,
                        force_pos=force_pos,
                        )
                    n += 1

                    acc += np.array_equal(p_word, g_word)

                    for attr, (i, j) in data.label_indices.items():
                        task_acc[attr] += np.array_equal(
                            p_word[i:j],
                            g_word[i:j],
                            )

        # print accuracies
        headers = ['ALL-OR-NOTHING', round(acc / n, 5)]
        table = [[t.upper(), round(acc / n, 5)] for t, acc in task_acc.items()]
        print(
            '\n\033[1mPrediction accuracies given %d words' % n,
            'across %d sentences\033[0m:' % len(predictions),
            )
        print(tabulate(table, headers=headers, tablefmt='fancy_grid'))

    @staticmethod
    def pos_confusion(predictions, gold, threshold=0.5, force_pos=True):
        n_tags = len(data.idx_attr['pos'])  # make extensible
        confusion = np.zeros((n_tags, n_tags))

        for p_sent, g_sent in zip(predictions, gold):
            for p_word, g_word in zip(p_sent, g_sent):
                p_word = list(data.binarize(
                    p_word,
                    threshold=threshold,
                    force_pos=force_pos,
                    ))[:n_tags]  # TODO
                g_word = list(g_word)[:n_tags]

                try:
                    confusion[g_word.index(1), p_word.index(1)] += 1
                except Exception:
                    pass

        # print POS-tagging confusion matrix
        headers = list(data.idx_attr['pos'].values())
        table = [[headers[i]] + list(confusion[i, :]) for i in range(n_tags)]
        headers = ['\n'.join(list(tag)) for tag in headers]
        print(
            '\n\033[1mPart-of-speech confusion matrix\033[0m:',
            '\nROW is the correct label, COLUMN is the predicted label.',
            )
        print(tabulate(table, headers=[''] + headers, tablefmt='fancy_grid'))


# playground ------------------------------------------------------------------

if __name__ == '__main__':

    # morphological tagger
    model = MorphTagger(
        model_fn=MODEL_FN,
        train_X=data.train_X,
        train_Y=data.train_Y,
        )
    predictions = model.predict(data.dev_X)
    model.acc(predictions, data.dev_Y)
    model.pos_confusion(predictions, data.dev_Y)
