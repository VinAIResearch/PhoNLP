# -*- coding: utf-8 -*-
"""
A trainer class to handle training and testing of models.
"""

from models.common.crf import viterbi_decode
from models.ner.vocab import MultiVocab
from models.ner.model import NERTagger
from models.common import utils
from models.common.trainer import Trainer as BaseTrainer
import sys
import logging
import torch
from torch import nn
import sys
sys.path.append('../')

logger = logging.getLogger('PhoNLPToolkit')


def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:8]]
    else:
        inputs = batch[:8]
    orig_idx = batch[8]
    word_orig_idx = batch[9]
    char_orig_idx = batch[10]
    sentlens = batch[11]
    wordlens = batch[12]
    charlens = batch[13]
    charoffsets = batch[14]
    return inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets


class TrainerNER(BaseTrainer):
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, model_file=None, config_phobert=None, use_cuda=False):
        self.use_cuda = use_cuda
        self.config_phobert = config_phobert
        if model_file is not None:
            # load everything from file
            self.load(model_file, args)
        else:
            assert all(var is not None for var in [args, vocab])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = NERTagger(args, vocab, self.config_phobert)
        #self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        #self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], momentum=self.args['momentum'])

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(
            batch, self.use_cuda)
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, chars, tags = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            # self.optimizer.zero_grad()
        loss, _ = self.model(tokens_phobert, words_phobert, word_mask, sentlens, eval=False, tags=tags)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        # self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(
            batch, self.use_cuda)
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, chars, tags = inputs

        self.model.eval()
        batch_size = word.size(0)
        logits = self.model(tokens_phobert, words_phobert, word_mask, sentlens, eval=True, tags=None)

        # decode
        trans = self.model.crit._transitions.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
            tags = self.vocab['tag'].unmap(tags)
            tag_seqs += [tags]

        if unsort:
            tag_seqs = utils.unsort(tag_seqs, orig_idx)
        return tag_seqs

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
            'model': model_state,
            'vocab': self.vocab.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, args=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint['config']
        if args:
            self.args.update(args)
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = NERTagger(self.args, self.vocab, self.config_phobert)
        self.model.load_state_dict(checkpoint['model'], strict=False)
