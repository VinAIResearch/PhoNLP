"""
A trainer class to handle training and testing of models.
"""

import sys

import torch
from phonlp.models.common import utils as util
from phonlp.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from phonlp.models.common.crf import viterbi_decode
from phonlp.models.common.trainer import Trainer as BaseTrainer
from phonlp.models.jointmodel.model import JointModel
from phonlp.models.ner.vocab import MultiVocab


def unpack_batch(batch, use_cuda, type):
    """ Unpack a batch from the data loader. """
    if type == "dep":
        if use_cuda:
            inputs = [b.cuda() if b is not None else None for b in batch[:5]]
        else:
            inputs = batch[:5]
        number_of_words = batch[5]
        orig_idx = batch[6]
        sentlens = batch[7]
        return inputs, number_of_words, orig_idx, sentlens
    elif type == "pos":
        if use_cuda:
            inputs = [b.cuda() if b is not None else None for b in batch[:3]]
        else:
            inputs = batch[:3]
        orig_idx = batch[3]
        sentlens = batch[4]
    elif type == "ner":
        if use_cuda:
            inputs = [b.cuda() if b is not None else None for b in batch[:4]]
        else:
            inputs = batch[:4]
        orig_idx = batch[4]
        sentlens = batch[5]
    return inputs, orig_idx, sentlens


class JointTrainer(BaseTrainer):
    """ A trainer for training models. """

    def __init__(self, args=None, vocab=None, model_file=None, config_phobert=None, use_cuda=False):
        self.use_cuda = use_cuda
        self.config_phobert = config_phobert
        if model_file is not None:
            self.load(model_file)
        else:
            self.args = args
            self.vocab = vocab
            self.model = JointModel(args, vocab, self.config_phobert)

        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()

    def update(self, batch_dep=None, batch_pos=None, batch_ner=None, lambda_pos=1.0, lambda_ner=1.0, lambda_dep=1.0):
        dep_inputs, dep_number_of_words, dep_orig_idx, dep_sentlens = unpack_batch(
            batch_dep, self.use_cuda, type="dep"
        )
        dep_tokens_phobert, dep_first_subword, dep_words_mask, dep_head, dep_deprel = dep_inputs
        pos_inputs, pos_orig_idx, pos_sentlens = unpack_batch(batch_pos, self.use_cuda, type="pos")
        pos_tokens_phobert, pos_first_subword, pos_upos = pos_inputs
        ner_inputs, ner_orig_idx, ner_sentlens = unpack_batch(batch_ner, self.use_cuda, type="ner")
        ner_tokens_phobert, ner_first_subword, ner_words_mask, ner_tags = ner_inputs

        self.model.train()
        loss_pos, loss_ner, loss_dep, loss, _ = self.model.forward(
            dep_tokens_phobert,
            dep_first_subword,
            dep_words_mask,
            dep_number_of_words,
            dep_head,
            dep_deprel,
            dep_sentlens,
            pos_tokens_phobert,
            pos_first_subword,
            pos_upos,
            pos_sentlens,
            ner_tokens_phobert,
            ner_first_subword,
            ner_words_mask,
            ner_sentlens,
            ner_tags,
            lambda_pos,
            lambda_ner,
            lambda_dep,
        )
        loss_val = loss.data.item()
        loss_val_pos = loss_pos.data.item()
        loss_val_ner = loss_ner.data.item()
        loss_val_dep = loss_dep.data.item()
        loss.backward()
        return loss_val, loss_val_pos, loss_val_ner, loss_val_dep

    def predict_dep(self, batch, unsort=True):
        dep_inputs, dep_number_of_words, dep_orig_idx, dep_sentlens = unpack_batch(batch, self.use_cuda, type="dep")
        dep_tokens_phobert, dep_first_subword, dep_words_mask, dep_head, dep_deprel = dep_inputs
        self.model.eval()
        batch_size = dep_tokens_phobert.size(0)
        loss_dep, preds = self.model.dep_forward(
            dep_tokens_phobert,
            dep_first_subword,
            dep_words_mask,
            dep_number_of_words,
            dep_sentlens,
            dep_head,
            dep_deprel,
            eval=True,
        )
        # dependency
        head_seqs = [
            chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], dep_sentlens)
        ]  # remove attachment for the root
        deprel_seqs = [
            self.vocab["deprel"].unmap([preds[1][i][j + 1][h] for j, h in enumerate(hs)])
            for i, hs in enumerate(head_seqs)
        ]
        pred_tokens = [
            [[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(dep_sentlens[i] - 1)] for i in range(batch_size)
        ]
        if unsort:
            pred_tokens = util.unsort(pred_tokens, dep_orig_idx)
        return pred_tokens

    def predict_pos(self, batch, unsort=True):
        pos_inputs, pos_orig_idx, pos_sentlens = unpack_batch(batch, self.use_cuda, type="pos")
        pos_tokens_phobert, pos_first_subword, pos_upos = pos_inputs
        self.model.eval()
        batch_size = pos_tokens_phobert.size(0)
        _, preds = self.model.pos_forward(pos_tokens_phobert, pos_first_subword, pos_sentlens, False, pos_upos)
        upos_seqs = [self.vocab["upos"].unmap(sent) for sent in preds[0].tolist()]

        pred_tokens = [
            [[upos_seqs[i][j]] for j in range(pos_sentlens[i])] for i in range(batch_size)
        ]  # , xpos_seqs[i][j], feats_seqs[i][j]
        if unsort:
            pred_tokens = util.unsort(pred_tokens, pos_orig_idx)
        return pred_tokens

    def predict_ner(self, batch, unsort=True):
        ner_inputs, ner_orig_idx, ner_sentlens = unpack_batch(batch, self.use_cuda, type="ner")
        ner_tokens_phobert, ner_first_subword, ner_word_mask, ner_tags = ner_inputs

        self.model.eval()
        loss, logits = self.model.ner_forward(
            ner_tokens_phobert, ner_first_subword, ner_word_mask, ner_sentlens, ner_tags
        )

        # decode
        trans = self.model.crit_ner._transitions.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, : ner_sentlens[i]], trans)
            tags = self.vocab["ner_tag"].unmap(tags)
            tag_seqs += [tags]
        if unsort:
            tag_seqs = util.unsort(tag_seqs, ner_orig_idx)
        return tag_seqs

    def save(self, filename):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        params = {"model": model_state, "vocab": self.vocab.state_dict(), "config": self.args}
        try:
            torch.save(params, filename)
            print("Model saved to {}".format(filename))
        except BaseException:
            print("Saving failed... continuing anyway.")

    def load(self, filename):
        """
        Load a model from file
        """
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            sys.exit(1)
        self.args = checkpoint["config"]
        self.vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
        # load model
        self.model = JointModel(self.args, self.vocab, config=self.config_phobert)
        self.model.load_state_dict(checkpoint["model"], strict=False)
