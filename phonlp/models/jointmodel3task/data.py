# -*- coding: utf-8 -*-
from models.ner.utils import is_bio_scheme, to_bio2, bio2_to_bioes
from models.ner.vocab import TagVocab, MultiVocab
from models.common.doc import *
from models.pos.vocab import CharVocab, WordVocab, MultiVocab
from models.common.vocab import PAD_ID, VOCAB_PREFIX, ROOT_ID, CompositeVocab
from models.common.data import map_to_ids, get_long_tensor, get_float_tensor, sort_all
import random
import logging
import torch

import sys
sys.path.append('../')
#from models.pos.xpos_vocab_factory import xpos_vocab_factory
logger = logging.getLogger('PhoNLPToolkit')


class DataLoaderPOS:
    def __init__(self, path_file, batch_size, args, vocab=None, evaluation=False, sort_during_eval=False,
                 tokenizer=None, max_seq_length=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval
        # self.doc = read_file(path_file)

        data = read_file(path_file)
        self.upos = [[w[1] for w in sent] for sent in data]
        # handle vocab
        self.vocab = vocab

        # handle pretrain; pretrain vocab is used when args['pretrain'] == True and pretrain is not None
        # self.pretrain_vocab = None
        # if pretrain is not None and args['pretrain']:
        #     self.pretrain_vocab = pretrain.vocab

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

            # if pretrain_vocab is not None:
            #     # always use lowercase lookup in pretrained vocab
            #     processed_sent += [pretrain_vocab.map([w[0].lower() for w in sent])]
            # else:
            #     processed_sent += [[PAD_ID] * len(sent)]
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
        # pretrained = get_long_tensor(batch[5], batch_size)
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
        self.head = [[w[1] for w in sent] for sent in data_dep]
        self.deprel = [[w[2] for w in sent] for sent in data_dep]

        # handle vocab
        self.vocab = vocab

        # handle pretrain; pretrain vocab is used when args['pretrain'] == True and pretrain is not None

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
            ##
            processed_sent += [[to_int(w[1], ignore_error=self.eval) for w in sent]]
            processed_sent += [vocab['deprel'].map([w[2] for w in sent])]
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

        assert len(dep_batch) == 6

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

        # dep_upos = get_long_tensor(dep_batch[4], dep_batch_size)
        # dep_pretrained = get_long_tensor(dep_batch[5], dep_batch_size)
        dep_sentlens = [len(x) for x in dep_batch[2]]
        dep_head = get_long_tensor(dep_batch[4], dep_batch_size)
        dep_deprel = get_long_tensor(dep_batch[5], dep_batch_size)
        dep_data = (dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars,
                    dep_wordchars_mask, dep_head, dep_deprel, dep_orig_idx, dep_word_orig_idx, dep_sentlens, dep_word_lens)
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
        data = doc.get([TEXT, HEAD, DEPREL], as_sentences=True)  # [[[cac value trong 1 token]]]
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
            (data,), self.data_orig_idx_dep = sort_all([data], [len(x[2]) for x in data])
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


class DataLoaderNER:
    def __init__(self, path_file, batch_size, args, vocab=None, evaluation=False, tokenizer=None, max_seq_length=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        # self.doc = doc

        data = read_file(path_file)
        # if self.preprocess_tags: # preprocess tags
        #     data = self.process_tags(data)
        self.tags = [[w[1] for w in sent] for sent in data]
        # print(data)

        # handle vocab
        # self.pretrain = pretrain
        self.vocab = vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            logger.debug("Subsample training set with rate {:g}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, args, tokenizer, max_seq_length)
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        logger.debug("{} batches created.".format(len(self.data)))

    # def init_vocab(self, data):
    #     def from_model(model_filename):
    #         """ Try loading vocab from charLM model file. """
    #         state_dict = torch.load(model_filename, lambda storage, loc: storage)
    #         assert 'vocab' in state_dict, "Cannot find vocab in charLM model file."
    #         return state_dict['vocab']
    #
    #     if self.eval:
    #         raise Exception("Vocab must exist for evaluation.")
    #     if self.args['charlm']:
    #         charvocab = CharVocab.load_state_dict(from_model(self.args['charlm_forward_file']))
    #     else:
    #         charvocab = CharVocab(data, self.args['shorthand'])
    #     # wordvocab = self.pretrain.vocab
    #     wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
    #     tagvocab = TagVocab(data, self.args['shorthand'], idx=2)
    #     vocab = MultiVocab({'char': charvocab,
    #                         'word': wordvocab,
    #                         'tag': tagvocab})
    #     return vocab

    def preprocess(self, data, vocab, args, tokenizer, max_seq_length):
        processed = []
        sep_id = 2
        cls_id = 0
        pad_id = 1
        if args.get('lowercase', True):  # handle word case
            def case(x): return x.lower()
        else:
            def case(x): return x
        if args.get('char_lowercase', False):  # handle character case
            def char_case(x): return x.lower()
        else:
            def char_case(x): return x

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
            processed_sent += [vocab['word'].map([case(w[0]) for w in sent])]
            processed_sent += [[vocab['char'].map([char_case(x) for x in w[0]]) for w in sent]]
            processed_sent += [vocab['ner_tag'].map([w[1] for w in sent])]
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0:  # or key % len(self.data):
            raise IndexError
        key = key % len(self.data)
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5  # words: List[List[int]], chars: List[List[List[int]]], tags: List[List[int]]

        # sort sentences by lens for easy RNN operations
        sentlens = [len(x) for x in batch[2]]
        batch, orig_idx = sort_all(batch, sentlens)
        sentlens = [len(x) for x in batch[2]]

        # sort chars by lens for easy char-LM operations
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward, charlens = self.process_chars(
            batch[3])
        chars_sorted, char_orig_idx = sort_all(
            [chars_forward, chars_backward, charoffsets_forward, charoffsets_backward], charlens)
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward = chars_sorted
        charlens = [len(sent) for sent in chars_forward]

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[3] for w in sent]
        wordlens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], wordlens)
        batch_words = batch_words[0]
        wordlens = [len(x) for x in batch_words]

        # convert to tensors
        tokens_phobert = batch[0]
        tokens_phobert = get_long_tensor(tokens_phobert, batch_size, pad_id=1)
        words_phobert = batch[1]
        words_phobert = get_long_tensor(words_phobert, batch_size)

        words = get_long_tensor(batch[2], batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(wordlens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)
        chars_forward = get_long_tensor(chars_forward, batch_size, pad_id=self.vocab['char'].unit2id(' '))
        chars_backward = get_long_tensor(chars_backward, batch_size, pad_id=self.vocab['char'].unit2id(' '))
        chars = torch.cat([chars_forward.unsqueeze(0), chars_backward.unsqueeze(0)]
                          )  # padded forward and backward char idx
        # idx for forward and backward lm to get word representation
        charoffsets = [charoffsets_forward, charoffsets_backward]
        tags = get_long_tensor(batch[4], batch_size)
        # print(tags)
        return tokens_phobert, words_phobert, words, words_mask, wordchars, wordchars_mask, chars, tags, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    # def load_doc(self, doc):
    #     data = doc.get([TEXT, NER], as_sentences=True, from_token=True)
    #     if self.preprocess_tags: # preprocess tags
    #         data = self.process_tags(data)
    #     return data

    def process_tags(self, sentences):
        res = []
        # check if tag conversion is needed
        convert_to_bioes = False
        is_bio = is_bio_scheme([x[2] for sent in sentences for x in sent])
        if is_bio and self.args.get('scheme', 'bio').lower() == 'bioes':
            convert_to_bioes = True
            logger.debug("BIO tagging scheme found in input; converting into BIOES scheme...")
        # process tags
        for sent in sentences:
            words, pos, tags = zip(*sent)
            # NER field sanity checking
            if any([x is None or x == '_' for x in tags]):
                raise Exception("NER tag not found for some input data.")
            # first ensure BIO2 scheme
            tags = to_bio2(tags)
            # then convert to BIOES
            if convert_to_bioes:
                tags = bio2_to_bioes(tags)
            res.append([[w, t] for w, t in zip(words, tags)])
        return res

    def process_chars(self, sents):
        start_id, end_id = self.vocab['char'].unit2id('\n'), self.vocab['char'].unit2id(' ')  # special token
        start_offset, end_offset = 1, 1
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward = [], [], [], []
        # get char representation for each sentence
        for sent in sents:
            chars_forward_sent, chars_backward_sent, charoffsets_forward_sent, charoffsets_backward_sent = [
                start_id], [start_id], [], []
            # forward lm
            for word in sent:
                chars_forward_sent += word
                charoffsets_forward_sent = charoffsets_forward_sent + \
                    [len(chars_forward_sent)]  # add each token offset in the last for forward lm
                chars_forward_sent += [end_id]
            # backward lm
            for word in sent[::-1]:
                chars_backward_sent += word[::-1]
                # add each offset in the first for backward lm
                charoffsets_backward_sent = [len(chars_backward_sent)] + charoffsets_backward_sent
                chars_backward_sent += [end_id]
            # store each sentence
            chars_forward.append(chars_forward_sent)
            chars_backward.append(chars_backward_sent)
            charoffsets_forward.append(charoffsets_forward_sent)
            charoffsets_backward.append(charoffsets_backward_sent)
        charlens = [len(sent) for sent in chars_forward]  # forward lm and backward lm should have the same lengths
        return chars_forward, chars_backward, charoffsets_forward, charoffsets_backward, charlens

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        return data


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


class BuildVocab:
    def __init__(self, args, path_pos, doc_dep, path_ner):
        self.args = args
        data_pos = read_file(path_pos)
        data_dep = self.load_doc_dep(doc_dep)
        data_ner = read_file(path_ner)
        self.vocab = self.build_vocab(data_dep, data_pos, data_ner)

    def build_vocab(self, data_dep, data_pos, data_ner):
        data = data_dep + data_pos + data_ner
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=7, lower=True)
        uposvocab = WordVocab(data_pos, self.args['shorthand'], idx=1)
        deprelvocab = WordVocab(data_dep, self.args['shorthand'], idx=2)
        ner_tag = TagVocab(data_ner, self.args['shorthand'], idx=1)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'deprel': deprelvocab,
                            'ner_tag': ner_tag})
        return vocab

    def load_doc_dep(self, doc):
        data = doc.get([TEXT, HEAD, DEPREL], as_sentences=True)  # [[[cac value trong 1 token]]]
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


def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res
