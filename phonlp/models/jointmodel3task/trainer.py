# -*- coding: utf-8 -*-
"""
A trainer class to handle training and testing of models.
"""

import sys
import logging
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from models.common.trainer import Trainer as BaseTrainer
from models.common import utils as util
from models.common.chuliu_edmonds import chuliu_edmonds_one_root
from models.jointmodel3task.model import *
# from PhoToolkit.models.pos.vocab import MultiVocab
from models.ner.vocab import MultiVocab
from models.common.crf import viterbi_decode

# logger = logging.getLogger('PhoToolkit')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def unpack_batch(batch, use_cuda, type):
    """ Unpack a batch from the data loader. """
    if type == 'dep':
        if use_cuda:
            inputs = [b.cuda() if b is not None else None for b in batch[:8]]
        else:
            inputs = batch[:8]
        orig_idx = batch[8]
        word_orig_idx = batch[9]
        sentlens = batch[10]
        wordlens = batch[11]
        return inputs, orig_idx, word_orig_idx, sentlens, wordlens
    elif type == 'pos':
        if use_cuda:
            inputs = [b.cuda() if b is not None else None for b in batch[:7]]
        else:
            inputs = batch[:7]
        orig_idx = batch[7]
        word_orig_idx = batch[8]
        sentlens = batch[9]
        wordlens = batch[10]
        return inputs, orig_idx, word_orig_idx, sentlens, wordlens
    elif type == 'ner':
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


class TrainerJoint(BaseTrainer):
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
            #self.model = JointModel(args, vocab, self.config_phobert, pretrain.emb if pretrain is not None else None, share_hid=False)
            self.model = JointModel(args, vocab, self.config_phobert)

        # self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        # self.optimizer = util.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], momentum=self.args['momentum'])

    def update(self, batch_dep=None, batch_pos=None, batch_ner=None, eval=False, lambda_pos=1.0, lambda_ner=1.0, lambda_dep=1.0):
        if batch_dep is not None:
            dep_inputs, dep_orig_idx, dep_word_orig_idx, dep_sentlens, dep_wordlens = unpack_batch(
                batch_dep, self.use_cuda, type='dep')
            dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars, dep_wordchars_mask, dep_head, dep_deprel = dep_inputs
        else:
            dep_inputs, dep_orig_idx, dep_word_orig_idx, dep_sentlens, dep_wordlens = 5 * [None]
            dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars, dep_wordchars_mask, dep_head, dep_deprel = 8 * \
                [None]
        if batch_pos is not None:
            pos_inputs, pos_orig_idx, pos_word_orig_idx, pos_sentlens, pos_wordlens = unpack_batch(
                batch_pos, self.use_cuda, type='pos')
            pos_tokens_phobert, pos_words_phobert, pos_words, pos_words_mask, pos_wordchars, pos_wordchars_mask, pos_upos = pos_inputs
        else:
            pos_inputs, pos_orig_idx, pos_word_orig_idx, pos_sentlens, pos_wordlens = 5 * [None]
            pos_tokens_phobert, pos_words_phobert, pos_words, pos_words_mask, pos_wordchars, pos_wordchars_mask, pos_upos = 7 * \
                [None]
        if batch_ner is not None:
            ner_inputs, ner_orig_idx, ner_word_orig_idx, ner_char_orig_idx, ner_sentlens, ner_wordlens, ner_charlens, ner_charoffsets = unpack_batch(
                batch_ner, self.use_cuda, type='ner')
            ner_tokens_phobert, ner_words_phobert, ner_words, ner_words_mask, ner_wordchars, ner_wordchars_mask, ner_chars, ner_tags = ner_inputs
        else:
            ner_inputs, ner_orig_idx, ner_word_orig_idx, ner_char_orig_idx, ner_sentlens, ner_wordlens, ner_charlens, ner_charoffsets = 8 * \
                [None]
            ner_tokens_phobert, ner_words_phobert, ner_words, ner_words_mask, ner_wordchars, ner_wordchars_mask, ner_chars, ner_tags = 8 * \
                [None]

        if eval:
            self.model.eval()
        else:
            self.model.train()
            # self.optimizer.zero_grad()

        loss_pos, loss_ner, loss_dep, loss, _ = self.model.forward(dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars, dep_head, dep_deprel, dep_sentlens,
                                                                   pos_tokens_phobert, pos_words_phobert, pos_upos, pos_sentlens,
                                                                   ner_tokens_phobert, ner_words_phobert, ner_words_mask, ner_sentlens, ner_tags,
                                                                   lambda_pos, lambda_ner, lambda_dep)
        loss_val = loss.data.item()
        loss_val_pos = loss_pos.data.item()
        loss_val_ner = loss_ner.data.item()
        loss_val_dep = loss_dep.data.item()
        if eval:
            return loss_val
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        # self.optimizer.step()
        return loss_val, loss_val_pos, loss_val_ner, loss_val_dep

    def predict_dep(self, batch, unsort=True, eval=True):
        dep_inputs, dep_orig_idx, dep_word_orig_idx, dep_sentlens, dep_wordlens = unpack_batch(batch, self.use_cuda,
                                                                                               type='dep')
        dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars, dep_wordchars_mask, dep_head, dep_deprel = dep_inputs
        self.model.eval()
        batch_size = dep_words.size(0)
        loss_dep, preds = self.model.dep_forward(
            dep_tokens_phobert, dep_words_phobert, dep_words, dep_words_mask, dep_wordchars, dep_sentlens, eval=eval)
        # dependency
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(
            preds[0], dep_sentlens)]  # remove attachment for the root
        deprel_seqs = [self.vocab['deprel'].unmap([preds[1][i][j+1][h]
                                                   for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]
        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]]
                        for j in range(dep_sentlens[i]-1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = util.unsort(pred_tokens, dep_orig_idx)

        return pred_tokens

    def predict_pos(self, batch, unsort=True, eval=True):
        pos_inputs, pos_orig_idx, pos_word_orig_idx, pos_sentlens, pos_wordlens = unpack_batch(batch, self.use_cuda,
                                                                                               type='pos')
        pos_tokens_phobert, pos_words_phobert, pos_words, pos_words_mask, pos_wordchars, pos_wordchars_mask, pos_upos = pos_inputs
        self.model.eval()
        batch_size = pos_words.size(0)
        preds, _ = self.model.pos_forward(pos_tokens_phobert, pos_words_phobert, pos_sentlens, eval, None)
        upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]

        pred_tokens = [[[upos_seqs[i][j]] for j in range(pos_sentlens[i])]
                       for i in range(batch_size)]  # , xpos_seqs[i][j], feats_seqs[i][j]
        if unsort:
            pred_tokens = util.unsort(pred_tokens, pos_orig_idx)
        return pred_tokens

    def predict_ner(self, batch, unsort=True, eval=True):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(
            batch, self.use_cuda, type='ner')
        tokens_phobert, words_phobert, word, word_mask, wordchars, wordchars_mask, chars, tags = inputs

        self.model.eval()
        batch_size = word.size(0)
        logits = self.model.ner_forward(tokens_phobert, words_phobert, word_mask, sentlens, eval=eval)

        # decode
        trans = self.model.crit_ner._transitions.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
            tags = self.vocab['ner_tag'].unmap(tags)
            tag_seqs += [tags]
        if unsort:
            tag_seqs = util.unsort(tag_seqs, orig_idx)
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
        self.model = JointModel(self.args, self.vocab, config=self.config_phobert)
        self.model.load_state_dict(checkpoint['model'], strict=False)
