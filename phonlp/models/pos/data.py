# -*- coding: utf-8 -*-
import random
import logging
import torch

from models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from models.common.vocab import PAD_ID, VOCAB_PREFIX
from models.pos.vocab import CharVocab, WordVocab, MultiVocab
# from models.common.doc import *

logger = logging.getLogger('PhoToolkit')


class DataLoaderPOS:
    def __init__(self, path_file, batch_size, args, vocab=None, evaluation=False, sort_during_eval=False, tokenizer=None, max_seq_length=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval

        data = read_file(path_file)
        self.upos = [[w[1] for w in sent] for sent in data]
        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, tokenizer, max_seq_length)
        print("NUMBER OF EXAMPLE IN DEP DATASET: ", len(data))
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        print("{} batches created.".format(len(self.data)))

    def init_vocab(self, data):
        assert self.eval == False  # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab
                            })
        return vocab

    def preprocess(self, data, vocab, tokenizer, max_seq_length):
        pad_id = 1
        cls_id = 0
        sep_id = 2
        processed = []
        for sent in data:
            input_ids = [cls_id]
            firstSWindices = [len(input_ids)]
            for w in sent:
                word_token = tokenizer.encode(w[0])
                input_ids += word_token[1:(len(word_token) - 1)]
                firstSWindices.append(len(input_ids))
            firstSWindices = firstSWindices[:(len(firstSWindices) - 1)]
            input_ids.append(sep_id)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
                # input_ids[-1] = eos_id
            else:
                input_ids = input_ids + [pad_id, ] * (max_seq_length - len(input_ids))

            processed_sent = [input_ids]
            processed_sent += [firstSWindices]
            processed_sent += [vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['upos'].map([w[1] for w in sent])]

            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        # if key < 0 or key >= len(self.data):
        #     raise IndexError
        if key > 0 and key % len(self.data) == 0:
            self.reshuffle()
        batch_key = key % len(self.data)
        batch = self.data[batch_key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[2]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[3] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        tokens_phobert = batch[0]
        tokens_phobert = get_long_tensor(tokens_phobert, batch_size, pad_id=1)
        words_phobert = batch[1]
        words_phobert = get_long_tensor(words_phobert, batch_size)
        words = batch[2]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_long_tensor(batch[4], batch_size)
        sentlens = [len(x) for x in batch[2]]
        return tokens_phobert, words_phobert, words, words_mask, wordchars, wordchars_mask, upos, orig_idx, word_orig_idx, sentlens, word_lens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    # def load_doc(self, doc):
    #     data = doc.get([TEXT, UPOS, XPOS, FEATS], as_sentences=True)
    #     data = self.resolve_none(data)
    #     return data

    def resolve_none(self, data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = '_'
        return data

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(self.data)
        self.data = self.chunk_batches(data)
        # random.shuffle(self.data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[2]), reverse=random.random() > .5)
        elif self.sort_during_eval:
            (data,), self.data_orig_idx_pos = sort_all([data], [len(x[2]) for x in data])

        current = []
        for x in data:
            if len(current) == self.batch_size:
                res.append(current)
                current = []
            current.append(x)

        if len(current) > 0:
            res.append(current)
        # elif len(current) > 0 and len(current) < self.batch_size:
        #     current += res[0][:(self.batch_size - len(current))]
        #     res.append(current)

        return res


def read_file(file):
    f = open(file)
    doc, sent = [], []
    for line in f:
        line = line.strip()
        if len(line) == 0:
            if len(sent) > 0:
                doc.append(sent)
                sent = []
        else:
            array = line.split('\t')
            sent += [array]
    if len(sent) > 0:
        doc.append(sent)
    return doc
