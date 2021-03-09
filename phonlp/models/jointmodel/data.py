import logging
import random

import torch
from phonlp.models.common.data import get_long_tensor, sort_all
from phonlp.models.common.doc import DEPREL, HEAD, TEXT
from phonlp.models.common.vocab import PAD_ID, ROOT_ID
from phonlp.models.ner.vocab import TagVocab
from phonlp.models.pos.vocab import CharVocab, MultiVocab, WordVocab


logger = logging.getLogger("PhoNLPToolkit")


class DataLoaderPOS:
    def __init__(
        self,
        path_file,
        batch_size,
        args,
        vocab=None,
        evaluation=False,
        sort_during_eval=False,
        tokenizer=None,
        max_seq_length=None,
    ):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval

        data = read_file(path_file)
        self.upos = [[w[1] for w in sent] for sent in data]
        # handle vocab
        self.vocab = vocab

        data = self.preprocess(data, self.vocab, tokenizer, max_seq_length)
        if evaluation is True:
            print("Number of evaluation sentences for POS tagging: ", len(data))
        else:
            print("Number of training sentences for POS tagging: ", len(data))
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)

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
                input_ids += word_token[1 : (len(word_token) - 1)]
                firstSWindices.append(len(input_ids))
            firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
            input_ids.append(sep_id)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = (
                    input_ids
                    + [
                        pad_id,
                    ]
                    * (max_seq_length - len(input_ids))
                )

            processed_sent = [input_ids]
            processed_sent += [firstSWindices]
            processed_sent += [vocab["upos"].map([w[1] for w in sent])]
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key > 0 and key % len(self.data) == 0:
            self.reshuffle()
        batch_key = key % len(self.data)
        batch = self.data[batch_key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 3

        lens = [len(x) for x in batch[2]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        tokens_phobert = batch[0]
        tokens_phobert = get_long_tensor(tokens_phobert, batch_size, pad_id=1)
        first_subword = batch[1]
        first_subword = get_long_tensor(first_subword, batch_size)
        upos = get_long_tensor(batch[2], batch_size)
        sentlens = [len(x) for x in batch[1]]
        return tokens_phobert, first_subword, upos, orig_idx, sentlens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def resolve_none(self, data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = "_"
        return data

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(self.data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[2]), reverse=random.random() > 0.5)
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

        return res


class DataLoaderDep:
    def __init__(
        self,
        doc_dep,
        batch_size,
        args,
        vocab=None,
        evaluation=False,
        sort_during_eval=False,
        tokenizer=None,
        max_seq_length=None,
    ):
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

        data_dep = self.preprocess(data_dep, self.vocab, tokenizer, max_seq_length)
        if evaluation is True:
            print("Number of evaluation sentences for dependency parsing: ", len(data_dep))
        else:
            print("Number of training sentences for dependency parsing: ", len(data_dep))

        # shuffle for training
        if self.shuffled:
            random.shuffle(data_dep)
        # chunk into batches
        self.data_dep = self.chunk_batches(data_dep)

    def preprocess(self, data_dep, vocab, tokenizer, max_seq_length):
        pad_id = 1
        cls_id = 0
        sep_id = 2
        processed_dep = []
        for sent in data_dep:
            input_ids = [cls_id]
            firstSWindices = [len(input_ids)]
            root_token = tokenizer.encode("[ROOT]")
            input_ids += root_token[1 : (len(root_token) - 1)]
            firstSWindices.append(len(input_ids))
            for w in sent:
                word_token = tokenizer.encode(w[0])
                input_ids += word_token[1 : (len(word_token) - 1)]
                firstSWindices.append(len(input_ids))
            firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
            input_ids.append(sep_id)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = (
                    input_ids
                    + [
                        pad_id,
                    ]
                    * (max_seq_length - len(input_ids))
                )

            processed_sent = [input_ids]
            processed_sent += [firstSWindices]
            processed_sent += [[ROOT_ID] + vocab["word"].map([w[0] for w in sent])]
            ##
            processed_sent += [[[ROOT_ID]] + [vocab["char"].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[to_int(w[1], ignore_error=self.eval) for w in sent]]
            processed_sent += [vocab["deprel"].map([w[2] for w in sent])]
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

        dep_lens = [len(x) for x in dep_batch[2]]
        dep_batch, dep_orig_idx = sort_all(dep_batch, dep_lens)

        dep_batch_words = [w for sent in dep_batch[3] for w in sent]
        dep_word_lens = [len(x) for x in dep_batch_words]
        dep_batch_words, dep_word_orig_idx = sort_all([dep_batch_words], dep_word_lens)
        dep_batch_words = dep_batch_words[0]  # [word1,...], word1 = list of tokens
        dep_word_lens = [len(x) for x in dep_batch_words]
        dep_wordchars = get_long_tensor(dep_batch_words, len(dep_word_lens))
        dep_number_of_words = dep_wordchars.size(0)
        dep_words = dep_batch[2]
        dep_words = get_long_tensor(dep_words, dep_batch_size)
        dep_words_mask = torch.eq(dep_words, PAD_ID)

        # convert to tensors
        dep_tokens_phobert = dep_batch[0]
        dep_tokens_phobert = get_long_tensor(dep_tokens_phobert, dep_batch_size, pad_id=1)
        dep_first_subword = dep_batch[1]
        dep_first_subword = get_long_tensor(dep_first_subword, dep_batch_size)
        dep_sentlens = [len(x) for x in dep_batch[1]]
        dep_head = get_long_tensor(dep_batch[4], dep_batch_size)
        dep_deprel = get_long_tensor(dep_batch[5], dep_batch_size)
        dep_data = (
            dep_tokens_phobert,
            dep_first_subword,
            dep_words_mask,
            dep_head,
            dep_deprel,
            dep_number_of_words,
            dep_orig_idx,
            dep_sentlens,
        )
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
                        data[sent_idx][tok_idx][feat_idx] = "_"
        return data

    def chunk_batches(self, data):
        res = []
        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[2]), reverse=random.random() > 0.5)
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
        return res  # list of list by sentences of list by token


class DataLoaderNER:
    def __init__(self, path_file, batch_size, args, vocab=None, evaluation=False, tokenizer=None, max_seq_length=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval

        data = read_file(path_file)
        self.tags = [[w[1] for w in sent] for sent in data]
        self.vocab = vocab

        data = self.preprocess(data, self.vocab, args, tokenizer, max_seq_length)
        if evaluation is True:
            print("Number of evaluation sentences for NER: ", len(data))
        else:
            print("Number of training sentences for NER: ", len(data))
        # shuffle for training
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)

    def preprocess(self, data, vocab, args, tokenizer, max_seq_length):
        processed = []
        sep_id = 2
        cls_id = 0
        pad_id = 1
        for sent in data:
            input_ids = [cls_id]
            firstSWindices = [len(input_ids)]
            for w in sent:
                word_token = tokenizer.encode(w[0])
                input_ids += word_token[1 : (len(word_token) - 1)]
                firstSWindices.append(len(input_ids))
            firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
            input_ids.append(sep_id)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
            else:
                input_ids = (
                    input_ids
                    + [
                        pad_id,
                    ]
                    * (max_seq_length - len(input_ids))
                )
            processed_sent = [input_ids]
            processed_sent += [firstSWindices]
            processed_sent += [vocab["word"].map([w[0] for w in sent])]
            processed_sent += [vocab["ner_tag"].map([w[1] for w in sent])]
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
        assert len(batch) == 4

        # sort sentences by lens for easy RNN operations
        sentlens = [len(x) for x in batch[2]]
        batch, orig_idx = sort_all(batch, sentlens)
        sentlens = [len(x) for x in batch[2]]

        words = get_long_tensor(batch[2], batch_size)
        words_mask = torch.eq(words, PAD_ID)
        # convert to tensors
        tokens_phobert = batch[0]
        tokens_phobert = get_long_tensor(tokens_phobert, batch_size, pad_id=1)
        first_subword = batch[1]
        first_subword = get_long_tensor(first_subword, batch_size)
        tags = get_long_tensor(batch[3], batch_size)
        return tokens_phobert, first_subword, words_mask, tags, orig_idx, sentlens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        data = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
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
            array = line.split("\t")
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
        uposvocab = WordVocab(data_pos, idx=1)
        deprelvocab = WordVocab(data_dep, idx=2)
        ner_tag = TagVocab(data_ner, idx=1)
        charvocab = CharVocab(data)
        wordvocab = WordVocab(data, cutoff=7, lower=True)
        vocab = MultiVocab(
            {"upos": uposvocab, "deprel": deprelvocab, "ner_tag": ner_tag, "char": charvocab, "word": wordvocab}
        )
        return vocab

    def load_doc_dep(self, doc):
        data = doc.get([TEXT, HEAD, DEPREL], as_sentences=True)
        data = self.resolve_none(data)
        return data

    def resolve_none(self, data):
        # replace None to '_'
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = "_"
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
