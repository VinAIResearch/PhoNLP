# -*- coding: utf-8 -*-
from transformers import *
from models.common.dropout import WordDropout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
import sys
sys.path.append('../')
# from PhoToolkit.models.common.vocab import CompositeVocab
# from PhoToolkit.models.common.char_model import CharacterModel
# from transformers import AutoModel


class Tagger(BertPreTrainedModel):
    def __init__(self, args, vocab, config):
        super(Tagger, self).__init__(config)

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []
        self.config = config

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        # if self.args['char'] and self.args['char_emb_dim'] > 0:
        #     self.charmodel = CharacterModel(args, vocab)
        #     self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
        #     input_size += self.args['transformed_dim']

        # if self.args['pretrain']:
        #     # pretrained embeddings, by default this won't be saved into model file
        #     add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
        #     self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
        #     input_size += self.args['transformed_dim']

        if self.args['use_phobert']:
            self.phobert = AutoModel.from_pretrained(self.args['phobert_model'], config=self.config)
            # self.trans_phobert = nn.Linear(self.config.hidden_size, self.args['transformed_dim'])
            input_size += self.config.to_dict()['hidden_size']

        # recurrent layers
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        # classifiers
        self.upos_hid = nn.Linear(self.config.to_dict()['hidden_size'], self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=1)  # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, tokens_phobert, words_phobert, sentlens, eval=False, upos=None):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['use_phobert']:
            # with torch.no_grad():
            phobert_emb = self.phobert(tokens_phobert)
            # print(len(phobert_emb))
            # print(phobert_emb[0].size())
            # print(phobert_emb[1].size())
            # print(phobert_emb)
            phobert_emb = phobert_emb[2][-1]
            phobert_emb = torch.cat([torch.index_select(phobert_emb[i], 0, words_phobert[i]).unsqueeze(0)
                                     for i in range(phobert_emb.size(0))], dim=0)
            phobert_emb = phobert_emb.cuda()
            phobert_emb = pack(phobert_emb)
            inputs += [phobert_emb]
        # if self.args['word_emb_dim'] > 0:
        #     word_emb = self.word_emb(word)
        #     word_emb = pack(word_emb)
        #     inputs += [word_emb]

        # if self.args['pretrain']:
        #     pretrained_emb = self.pretrained_emb(pretrained)
        #     pretrained_emb = self.trans_pretrained(pretrained_emb)
        #     pretrained_emb = pack(pretrained_emb)
        #     inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, phobert_emb.batch_sizes), batch_first=True)[0]

        # if self.args['char'] and self.args['char_emb_dim'] > 0:
        #     char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
        #     char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
        #     inputs += [char_reps]

        #lstm_inputs = torch.cat([x.data for x in inputs], 1)
        inputs = inputs[0].data
        ###
        inputs = self.worddrop(inputs, self.drop_replacement)

        ######
        upos_hid = F.relu(self.upos_hid(inputs))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        preds = [pad(upos_pred).max(2)[1]]

        if eval == False:
            upos = pack(upos).data
            loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
            return loss, preds
        else:
            return preds
