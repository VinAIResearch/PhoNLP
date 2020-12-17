# -*- coding: utf-8 -*-
import random
import logging
import torch

from models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
from models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab
from models.pos.vocab import CharVocab, WordVocab, MultiVocab
from models.common.doc import *

# from transformers.modeling_utils import *
# from fairseq.data.encoders.fastbpe import fastBPE
# from fairseq.data import Dictionary
# from transformers import *

logger = logging.getLogger('PhoNLPToolkit')


class DataLoaderDep:
    def __init__(self, doc_dep, batch_size, args, vocab=None, evaluation=False, sort_during_eval=False,
                 tokenizer=None, max_seq_length=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval
        self.doc_dep = doc_dep
        data_dep = self.load_doc(doc_dep)  # list sentences of list words of list fields
        self.head = [[w[2] for w in sent] for sent in data_dep]
        self.deprel = [[w[3] for w in sent] for sent in data_dep]

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data_dep)
        else:
            self.vocab = vocab

        # handle pretrain; pretrain vocab is used when args['pretrain'] == True and pretrain is not None
        # self.pretrain_vocab = None
        # if pretrain is not None and args['pretrain']:
        #     self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep_dep = int(args['sample_train'] * len(data_dep))
            data_dep = random.sample(data_dep, keep_dep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data_dep = self.preprocess(data_dep, self.vocab, tokenizer, max_seq_length)
        print("NUMBER OF EXAMPLE IN DEP DATASET: ", len(data_dep))

        # shuffle for training
        if self.shuffled:
            random.shuffle(data_dep)
        # chunk into batches
        self.data_dep = self.chunk_batches(data_dep)
        print("{} dep batches created.".format(len(self.data_dep)))

    def init_vocab(self, data):
        assert self.eval == False  # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        deprelvocab = WordVocab(data, self.args['shorthand'], idx=3)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'deprel': deprelvocab})
        return vocab

    def preprocess(self, data_dep, vocab, tokenizer, max_seq_length):
        pad_id = 1
        cls_id = 0
        sep_id = 2
        processed_dep = []
        for sent in data_dep:
            input_ids = [cls_id]
            firstSWindices = [len(input_ids)]
            root_token = tokenizer.encode("[ROOT]")
            input_ids += root_token[1:(len(root_token) - 1)]
            firstSWindices.append(len(input_ids))
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
            processed_sent += [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            # [[root_id], [l,i,n,h],..]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            ##
            # if pretrain_vocab is not None:
            #     # always use lowercase lookup in pretrained vocab
            #     processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0].lower() for w in sent])]
            # else:
            #     processed_sent += [[ROOT_ID] + [PAD_ID] * len(sent)]
            processed_sent += [[to_int(w[2], ignore_error=self.eval) for w in sent]]
            processed_sent += [vocab['deprel'].map([w[3] for w in sent])]
            processed_dep.append(processed_sent)

        return processed_dep  # [sent1,sent2,...] sent1 = list of list

    def __len__(self):
        return len(self.data_dep)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key > 0 and key % len(self.data_dep) == 0:
            self.reshuffle()
        dep_key = key % len(self.data_dep)
        dep_batch = self.data_dep[dep_key]
        dep_batch_size = len(dep_batch)
        dep_batch = list(zip(*dep_batch))

        assert len(dep_batch) == 7

        # sort sentences by lens for easy RNN operations
        dep_lens = [len(x) for x in dep_batch[2]]
        dep_batch, dep_orig_idx = sort_all(dep_batch, dep_lens)

        # sort words by lens for easy char-RNN operations
        dep_batch_words = [w for sent in dep_batch[3] for w in sent]
        dep_word_lens = [len(x) for x in dep_batch_words]
        dep_batch_words, dep_word_orig_idx = sort_all([dep_batch_words], dep_word_lens)
        dep_batch_words = dep_batch_words[0]  # [word1,...], word1 = list of tokens
        dep_word_lens = [len(x) for x in dep_batch_words]

        # convert to tensors
        dep_tokens_phobert = dep_batch[0]
        dep_tokens_phobert = get_long_tensor(dep_tokens_phobert, dep_batch_size, pad_id=1)
        dep_words_phobert = dep_batch[1]
        dep_words_phobert = get_long_tensor(dep_words_phobert, dep_batch_size)
        dep_words = dep_batch[2]  # [sen1,sent2,..] , sent1 = list of words
        dep_words = get_long_tensor(dep_words, dep_batch_size)  # batchsize * max_len_sent
        dep_words_mask = torch.eq(dep_words, PAD_ID)  # same size words, pad_id = 1, no =0
        dep_wordchars = get_long_tensor(dep_batch_words, len(dep_word_lens))  # number of  words * max_len_word
        dep_wordchars_mask = torch.eq(dep_wordchars, PAD_ID)

        dep_upos = get_long_tensor(dep_batch[4], dep_batch_size)
        # dep_pretrained = get_long_tensor(dep_batch[5], dep_batch_size)
        dep_sentlens = [len(x) for x in dep_batch[2]]
        dep_head = get_long_tensor(dep_batch[5], dep_batch_size)
        dep_deprel = get_long_tensor(dep_batch[6], dep_batch_size)
        dep_data = (dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars, dep_wordchars_mask,
                    dep_upos, dep_head, dep_deprel, dep_orig_idx, dep_word_orig_idx, dep_sentlens, dep_word_lens)
        return dep_data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data_dep = [y for x in self.data_dep for y in x]
        random.shuffle(self.data_dep)
        self.data_dep = self.chunk_batches(data_dep)
        random.shuffle(self.data_dep)

    def load_doc(self, doc):
        data = doc.get([TEXT, XPOS, HEAD, DEPREL], as_sentences=True)  # [[[cac value trong 1 token]]]
        data = self.resolve_none(data)
        return data

    def resolve_none(self, data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = '_'
        return data

    def chunk_batches(self, data):
        res = []
        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[2]), reverse=random.random() > .5)
        elif self.sort_during_eval:
            (data,), self.data_orig_idx = sort_all([data], [len(x[2]) for x in data])
        current = []
        for x in data:
            if len(current) >= self.batch_size:
                res.append(current)
                current = []
            current.append(x)

        if len(current) > 0:
            res.append(current)
        # if len(current) == self.batch_size:
        #     res.append(current)
        # elif len(current) > 0 and len(current) < self.batch_size:
        #     current += res[0][:(self.batch_size - len(current))]
        #     res.append(current)
        return res  # list of list by sentences of list by token


def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res
