# -*- coding: utf-8 -*-
"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
from torch import nn

from models.common.trainer import Trainer as BaseTrainer
from models.common import utils, loss
from models.common.chuliu_edmonds import chuliu_edmonds_one_root
from models.depparse.model import Parser
from models.pos.vocab import MultiVocab

logger = logging.getLogger('PhoNLPToolkit')


def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:9]]
    else:
        inputs = batch[:9]
    orig_idx = batch[9]
    word_orig_idx = batch[10]
    sentlens = batch[11]
    wordlens = batch[12]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens


class TrainerDep(BaseTrainer):
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
            self.model = Parser(args, vocab, self.config_phobert)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        # self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in self.parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # num_train_optimization_steps = int(self.args['num_epoch'] * num_batchsize / self.args['accumulation_steps']
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=args['lr'],
        #                   correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=100,
        #                                             num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        # self.scheduler0 = get_constant_schedule(optimizer)

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, upos, head, deprel = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            # self.optimizer.zero_grad()
        loss, _ = self.model(tokens_phobert, words_phobert, word, word_mask,
                             wordchars, upos, sentlens, head, deprel, eval=False)
        loss_val = loss.data.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        # self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, upos, head, deprel = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(tokens_phobert, words_phobert, word, word_mask, wordchars,
                              upos, sentlens, head=None, deprel=None, eval=True)

        # Upos
        # upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]
        # pred_tokens_upos = [[upos_seqs[i][j] for j in range(1, sentlens[i])] for i in range(batch_size)] #, xpos_seqs[i][j], feats_seqs[i][j]

        # dependency
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:]
                     for adj, l in zip(preds[0], sentlens)]  # remove attachment for the root
        deprel_seqs = [self.vocab['deprel'].unmap([preds[1][i][j+1][h]
                                                   for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]
        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens[i]-1)]
                       for i in range(batch_size)]
        if unsort:
            # preds_tokens_upos = utils.unsort(pred_tokens_upos, orig_idx)
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
        except BaseException:
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
        # emb_matrix = None
        # if self.args['pretrain'] and pretrain is not None: # we use pretrain only if args['pretrain'] == True and pretrain is not None
        #     emb_matrix = pretrain.emb
        self.model = Parser(self.args, self.vocab, config=self.config_phobert)
        self.model.load_state_dict(checkpoint['model'], strict=False)
