# -*- coding: utf-8 -*-
"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from models.common.trainer import Trainer as BaseTrainer
from models.common import utils
from models.pos.model import Tagger
from models.pos.vocab import MultiVocab

logger = logging.getLogger('PhoNLPToolkit')


def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:7]]
    else:
        inputs = batch[:7]
    orig_idx = batch[7]
    word_orig_idx = batch[8]
    sentlens = batch[9]
    wordlens = batch[10]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens


class TrainerPOS(BaseTrainer):
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, model_file=None, config_phobert=None, use_cuda=False):
        self.use_cuda = use_cuda
        self.config_phobert = config_phobert
        if model_file is not None:
            # load everything from file
            self.load(model_file)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Tagger(args, vocab, self.config_phobert)
        # self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        # self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, upos = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            # self.optimizer.zero_grad()
        loss, _ = self.model(tokens_phobert, words_phobert, sentlens, eval=False, upos=upos)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, upos = inputs

        self.model.eval()
        batch_size = word.size(0)
        preds = self.model(tokens_phobert, words_phobert, sentlens, eval=True)
        upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]
        pred_tokens = [[[upos_seqs[i][j]] for j in range(sentlens[i])]
                       for i in range(batch_size)]  # , xpos_seqs[i][j], feats_seqs[i][j]

        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

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

    def load(self, filename):
        """
        Load a model from file, with preloaded pretrain embeddings. Here we allow the pretrain to be None or a dummy input,
        and the actual use of pretrain embeddings will depend on the boolean config "pretrain" in the loaded args.
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.exception("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        # load model
        # if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
        #     emb_matrix = pretrain.emb
        self.model = Tagger(self.args, self.vocab, config=self.config_phobert)
        self.model.load_state_dict(checkpoint['model'], strict=False)
