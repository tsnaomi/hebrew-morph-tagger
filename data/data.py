'''
This file defines the `Data()` class. This class, when instantiated, sifts
through the HaAretz corpus, extracting sentences and token labels. It then
composes the training, development, and test datasets, providing useful tools
and information for interacting with the datasets.
'''

import numpy as np
import pickle
import os.path as op
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from random import shuffle

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

try:
    from .labels import attr_idx, idx_attr

except ImportError:
    from labels import attr_idx, idx_attr


XML_DIR = op.join(op.dirname(__file__), 'HaAretz/XML/')
DATA_DIR = op.join(op.dirname(__file__), 'HaAretz/')


class Data:

    def __init__(self, xml_dir=XML_DIR, data_dir=DATA_DIR, build=False):
        self.xml_dir = xml_dir
        self.data_dir = data_dir
        self.idx_attr = idx_attr
        self.dictionaries_fn = data_dir + 'dictionaries.dat'
        self.parameters_fn = data_dir + 'parameters.npy'

        # if `build` is True, generate the datasets from scratch...
        if build:
            self.create_datasets()

        # otherwise, try to load them from file...
        else:
            self.load()

    # datasets ----------------------------------------------------------------

    def load(self):
        '''Load datasets, dictionaries, and parameters from file.

        If any these are not found, generate everything from scratch.
        '''
        try:
            # load datasets
            self.train_X = self.unpickle(self.data_dir + 'train_X')
            self.train_Y = self.unpickle(self.data_dir + 'train_Y')
            self.dev_X = self.unpickle(self.data_dir + 'dev_X')
            self.dev_Y = self.unpickle(self.data_dir + 'dev_Y')
            self.test_X = self.unpickle(self.data_dir + 'test_X')
            self.test_Y = self.unpickle(self.data_dir + 'test_Y')

            # load dictionaries
            self.word_idx, self.idx_word, self.char_idx, self.label_indices = \
                Data.unpickle(self.dictionaries_fn)

            # load parameters
            self.max_sent_len, self.max_word_len, self.output_len, \
                self.n_words, self.n_chars = np.load(self.parameters_fn)

        except FileNotFoundError:
            print('Uh oh. Items missing from %s.' % self.data_dir)

            self.create_datasets()

    def create_datasets(self):
        '''Create 3 datasets: 2 input datasets and 1 target label dataset.

        Let N be the total number of sentences containing valid labels for each
        word in the sentence.

        1. WORDS is an N x `self.max_sent_len` matrix, containing word-level
           vector representations for each (valid) sentence. In each sentence
           vector, every word is represented by its index in `self.word_idx`.

        2. CHARS is an N x `self.max_sent_len` x `self.max_word_len` matrix,
           where each sentence is represented by a vector of vectors. Each word
           in a sentence is represented by a vector, in which every character
           in the word is represented by its index in `self.char_idx`.

        3. LABELS is an N x `self.output_len` matrix, where each row is the
            vectorized label for the sentence at the corresponding index in
            WORDS and CHARS.
        '''
        self.extract_vocab()
        sentences = self.extract_sentences()

        print('Creating datasets...')

        WORDS = []
        CHARS = []
        LABELS = []

        for sent in sentences:
            words = []
            chars = []
            labels = []
            n = 0

            for i, (word, label) in enumerate(sent):
                chars.append([self.char_idx[c] for c in word])
                words.append(self.word_idx[word])
                labels.append(label)
                n += 1

            chars = pad_sequences(
                maxlen=self.max_word_len,
                sequences=chars,
                value=0,
                padding='post',
                truncating='post',
                )

            WORDS.append(words)
            CHARS.append(chars)
            LABELS.append(labels)

        WORDS = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=WORDS,
            value=0,
            padding='post',
            truncating='post',
            )

        CHARS = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=CHARS,
            value=[0, ] * self.max_word_len,
            padding='post',
            truncating='post',
            )

        LABELS = pad_sequences(
            maxlen=self.max_sent_len,
            sequences=LABELS,
            value=[0, ] * self.output_len,
            padding='post',
            truncating='post',
            )

        # print('Sanity check:', WORDS.shape, CHARS.shape, LABELS.shape)

        self.split_and_save_datasets(WORDS, CHARS, LABELS)

    def split_and_save_datasets(self, words, chars, labels):
        '''Perform and save a random 80-10-10 split of the datasets.'''
        print('Splitting datasets...')

        N = len(labels)

        indices = [i for i in range(N)]
        shuffle(indices)
        indices = np.array(indices)

        # 80-10-10 split
        train_indices = indices[:round(N * .8)]
        end = len(train_indices)
        dev_indices = indices[end:end + round(N * .1)]
        end += len(dev_indices)
        test_indices = indices[end:]

        self.train_X = [words[train_indices, :], chars[train_indices, :]]
        self.dev_X = [words[dev_indices, :], chars[dev_indices, :]]
        self.test_X = [words[test_indices, :], chars[test_indices, :]]

        self.train_Y = labels[train_indices, :]
        self.dev_Y = labels[dev_indices, :]
        self.test_Y = labels[test_indices, :]

        # save training datasets
        self.pickle(self.data_dir + 'train_X', self.train_X)
        self.pickle(self.data_dir + 'train_Y', self.train_Y)

        # save dev datasets
        self.pickle(self.data_dir + 'dev_X', self.dev_X)
        self.pickle(self.data_dir + 'dev_Y', self.dev_Y)

        # save test datasets
        self.pickle(self.data_dir + 'test_X', self.test_X)
        self.pickle(self.data_dir + 'test_Y', self.test_Y)

        # save dictionaries and parameters too...
        Data.pickle(self.dictionaries_fn, (
            self.word_idx,
            self.idx_word,
            self.char_idx,
            self.label_indices,
            ))

        np.save(self.parameters_fn, np.array([
            self.max_sent_len,
            self.max_word_len,
            self.output_len,
            self.n_words,
            self.n_chars,
            ]))

        print('YAY! We have fresh datasets.')

    # labeling ----------------------------------------------------------------

    def get_label(self, tag):
        '''Vectorize the gold label for the token represented by `tag`.

        Note that `tag` is an XML element from the HaAretz corpus.
        '''
        #        pos
        label = [self.attr_vec['pos'][tag.tag.lower()], ]

        for attr in ['person', 'gender', 'number', 'tense', 'definiteness']:
            label.append(self.attr_vec[attr][tag.attrib.get(attr, None)])

        label = np.concatenate(label)

        return label

    def decipher(self, vec, threshold=0.5, force_pos=True):
        '''Return a human readable string for the label `vec`.

        The `threshold` argument determines the threshold for considering a
        label 'active'. For instance, if `threshold=0.5` and we have the value
        [0.43] at the index corresponding to 'adjective', then we would NOT
        consider the word labeled with `vec` to be an adjective. In contrast,
        if its value were [0.51], then we WOULD consider it an adjective.

        If `force_pos` is True, then the part-of-speech label with the highest
        value is interpreted as the part of speech, regardless of whether that
        value meets the `threshold` argument.
        '''
        vec = self.binarize(vec, threshold=threshold, force_pos=force_pos)
        label = []

        for attr, (i, j) in self.label_indices.items():
            partition = vec[i:j]

            for idx in np.where(partition > 0)[0]:
                label.append(self.idx_attr[attr][idx])

        label = ' '.join(label) \
            .replace('false', 'indefinite') \
            .replace('true', 'definite')

        return label or '<NO LABELS PREDICTED>'

    def binarize(self, vec, threshold=0.5, force_pos=True):
        '''Convert the values in `vec` to 1s and 0s.

        The `threshold` argument determines the threshold for changing a value
        to 1 or 0. If the value is greater or equal to `threshold`, the value
        is set to 1; otherwise, it is set to 0.

        If `force_pos` is True, then, for the portion of the label that
        determines part of speech, the highest value is set to 1, regardless of
        whether that value meets the `threshold` argument.
        '''
        new = np.array(vec)

        if force_pos:
            i, j = self.label_indices['pos']  # TODO: make extensible
            pos = new[i:j].argmax()
            new[i:j] = 0
            new[new >= threshold] = 1
            new[new < threshold] = 0
            new[pos] = 1

        else:
            new[new >= threshold] = 1
            new[new < threshold] = 0

        return new

    def get_sentence_input(self, tokenized_sent):
        '''Generate the input vectors for the sentence `tokenized_sent`.'''
        word_input, char_input = [], []

        for word in tokenized_sent:
            word_input.append(self.word_idx[word])
            chars = []

            for char in word:
                chars.append(self.char_idx[char])

            for _ in range(self.max_word_len - len(word)):
                chars.append(0)

            char_input.append(chars)

        pad = [0, ] * self.max_word_len

        for _ in range(self.max_sent_len - len(tokenized_sent)):
            word_input.append(0)
            char_input.append(pad)

        word_input = np.array([word_input, ])
        char_input = np.array([char_input, ])

        return [word_input, char_input]

    # metrics -----------------------------------------------------------------

    def corpus_metrics(self):
        '''Print counts for each label in the HaAretz corpus.

        This method was used to obtain the corpus counts listed in `labels.py`.
        '''
        tags = {'pos': [], }
        attributes = ['gender', 'number', 'person', 'tense', 'definiteness']
        tags.update({attr: [] for attr in attributes})

        for tree in self.xml_walk():
            for tok in tree.findall('.//token'):
                try:
                    analysis = tok.find(".//analysis[@score='1.0']")
                    tag = analysis.find(".//base")[0]
                    tags['pos'].append(tag.tag.lower())

                    for attr in attributes:
                        try:
                            tags[attr].append(tag.attrib[attr])

                        except KeyError:
                            tags[attr].append('n/a')

                except (AttributeError, TypeError):
                    continue

        tags = {k: dict(Counter(v)) for k, v in tags.items()}

        for tag, attrs in tags.items():
            print('%s:' % tag.upper())

            for attr, n in attrs.items():
                print('\t%s: %i' % (attr, n))

        print('\nn_words: ', self.n_words)()
        print('n_chars: ', self.n_chars)

    # extraction helpers ------------------------------------------------------

    def extract_vocab(self):
        '''Extract vocabulary-level information.

        This method is called in `self.create_datasets()`. It creates the
        dictionaries `self.idx_word`, `word_idx`, and `char_idx`. It also
        calculates the total number of unique words and characters in the
        corpus, as well as the maximum word length.
        '''
        print('Extracting vocabulary...')

        vocab = set()
        chars = set(' ')
        word_lengths = defaultdict(int)

        for tree in self.xml_walk():
            for tok in tree.findall('.//token'):
                word = tok.attrib['surface']
                word_lengths[len(word)] += 1
                vocab.add(word)
                chars.update(iter(word))

        self.word_idx = {'<PAD>': 0, }
        self.idx_word = {0: '<PAD>', }

        for idx, word in enumerate(vocab, start=1):
            self.word_idx[word] = idx
            self.idx_word[idx] = word

        self.char_idx = {'<PAD>': 0, }

        for idx, char in enumerate(chars, start=1):
            self.char_idx[char] = idx

        self.max_word_len = max(word_lengths.keys())
        self.n_words = len(self.word_idx)
        self.n_chars = len(self.char_idx)

    def extract_sentences(self):
        '''Extract sentence-level information.

        This method is called in `self.create_datasets()`. It returns the
        sentences contained in the HaAretz corpus. Each sentence is represented
        as a list of words, where each word is represented by a tuple. Each
        tuple consists of the word (in string form) and the word's vectorized
        label.
        '''
        self.vectorize_labels()

        print('Extracting sentences...')

        sentences = []

        for tree in self.xml_walk():
            for sentence in tree.findall('.//sentence'):
                try:
                    sent = []

                    for tok in sentence.findall('.//token'):
                        word = tok.attrib['surface']

                        try:
                            analysis = tok.find(".//analysis[@score='1.0']")
                            tag = analysis.find(".//base")[0]
                            sent.append((word, self.get_label(tag)))

                        except (AttributeError, TypeError, KeyError):
                            # for now, exclude any sentence that contains a
                            # word with an invalid label (as determined in
                            # `labels.py`)
                            raise ValueError

                except ValueError:
                    continue  # skip the sentence (3527 / 3989)

                sentences.append(sent)

        self.max_sent_len = max([len(s) for s in sentences])

        return sentences

    def vectorize_labels(self):
        '''Create vectorized labels based on `attr_idx`.

        This method is called in `self.extract_sentences()`. It creates the
        dictionary `self.attr_vec`, which houses the vectorized labels. It also
        creates `self.attr_ indices`, which contains the start and end indices
        for each attribute represented in a label vector (these indices are
        used in `self.decipher()`).
        '''
        print('Vectorizing labels...')

        self.attr_vec = {}
        self.label_indices = {}
        i = 0

        for attr, vals in attr_idx.items():
            one_hots = to_categorical(list(vals.values()))
            self.attr_vec[attr] = dict(zip(vals.keys(), one_hots))
            n = len(vals)
            j = i + n
            self.label_indices[attr] = (i, j)
            i = j

            # create a 'n/a' vector filled with zeros
            self.attr_vec[attr][None] = np.array([0, ] * n)

            self.output_len = j

        # for 'person', map 'any' to '1', '2', AND '3'
        self.attr_vec['person']['any'] = sum([
            self.attr_vec['person']['1'],
            self.attr_vec['person']['2'],
            self.attr_vec['person']['3'],
            ])

        # for 'gender', map 'masculine and feminine' to 'feminine' AND
        # 'masculine'
        self.attr_vec['gender']['masculine and feminine'] = sum([
            self.attr_vec['gender']['masculine'],
            self.attr_vec['gender']['feminine'],
            ])

        # for 'number', map 'dual and plural' to 'dual' AND 'plural', map
        # map 'dual' to 'dual' AND 'plural', and map singular and plural' to
        # 'singular' AND 'plural',
        self.attr_vec['number']['dual and plural'] = sum([
            self.attr_vec['number']['dual'],
            self.attr_vec['number']['plural'],
            ])
        self.attr_vec['number']['dual'] = \
            self.attr_vec['number']['dual and plural']

        self.attr_vec['number']['singular and plural'] = sum([
            self.attr_vec['number']['singular'],
            self.attr_vec['number']['plural'],
            ])

        # for 'tense', map 'bareInfinitive' to 'infinitive'
        self.attr_vec['tense']['bareInfinitive'] = \
            self.attr_vec['tense']['infinitive']

    # utilities ---------------------------------------------------------------

    def xml_walk(self):
        '''Yeild XML trees for each file in the HaAretz corpus.'''
        for i in range(1, 241):
            fn = self.xml_dir + '{:03d}.xml'.format(i)

            try:
                yield ET.parse(fn)

            except FileNotFoundError:
                continue

    @staticmethod
    def pickle(fn, data):
        '''Pickle `data` to `fn`.'''
        with open(fn, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def unpickle(fn):
        '''Load the contents of `fn`.'''
        with open(fn, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    data = Data(build=True)
