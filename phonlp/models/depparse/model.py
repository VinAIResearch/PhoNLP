# -*- coding: utf-8 -*-
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from models.common.biaffine import DeepBiaffineScorer
from models.common.dropout import WordDropout
from transformers import *
import sys
sys.path.append('../')


class Parser(BertPreTrainedModel):
    def __init__(self, args, vocab, config):  # , vocab_phobert, bpe, max_seq_length
        super(Parser, self).__init__(config)

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []
        # self.vocab_phobert = vocab_phobert
        self.config = config

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        # if self.args['word_emb_dim'] > 0:
        #     # frequent word embeddings
        #     self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
        # #     # self.lemma_emb = nn.Embedding(len(vocab['lemma']), self.args['word_emb_dim'], padding_idx=0)
        # #     # input_size += self.args['word_emb_dim'] * 2
        #     input_size = self.args['word_emb_dim']

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
            self.phobert = AutoModel.from_pretrained(args['phobert_model'], config=self.config)
            # self.trans_phobert = nn.Linear(self.config.hidden_size, self.args['transformed_dim'])
            input_size += self.config.to_dict()['hidden_size']

        if self.args['tag_emb_dim'] > 0:
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

            # if not isinstance(vocab['xpos'], CompositeVocab):
            #     self.xpos_emb = nn.Embedding(len(vocab['xpos']), self.args['tag_emb_dim'], padding_idx=0)
            # else:
            #     self.xpos_emb = nn.ModuleList()
            #
            #     for l in vocab['xpos'].lens():
            #         self.xpos_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))
            #
            # self.ufeats_emb = nn.ModuleList()
            #
            # for l in vocab['feats'].lens():
            #     self.ufeats_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))
            #
            # input_size += self.args['tag_emb_dim'] * 2
            input_size += self.args['tag_emb_dim']

        # recurrent layers
        # self.parserlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        # self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        # self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers upos
        self.dep_hid = nn.Linear(input_size, input_size)
        # self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        # self.upos_clf.weight.data.zero_()
        # self.upos_clf.bias.data.zero_()

        # classifiers
        self.unlabeled = DeepBiaffineScorer(
            input_size, input_size, self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        self.deprel = DeepBiaffineScorer(input_size, input_size, self.args['deep_biaff_hidden_dim'], len(
            vocab['deprel']), pairwise=True, dropout=args['dropout'])
        if args['linearization']:
            self.linearization = DeepBiaffineScorer(
                input_size, input_size, self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        if args['distance']:
            self.distance = DeepBiaffineScorer(
                input_size, input_size, self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')  # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, tokens_phobert, words_phobert, word, word_mask, wordchars, upos, sentlens, head=None, deprel=None, eval=False):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['use_phobert']:
            # with torch.no_grad():
            phobert_emb = self.phobert(tokens_phobert)
            phobert_emb = phobert_emb[2][-1]
            phobert_emb = torch.cat([torch.index_select(phobert_emb[i], 0, words_phobert[i]).unsqueeze(0)
                                     for i in range(phobert_emb.size(0))], dim=0)
            phobert_emb = phobert_emb.cuda()
            phobert_emb = pack(phobert_emb)
            inputs += [phobert_emb]
        # if self.args['pretrain']:
        #     pretrained_emb = self.pretrained_emb(pretrained)
        #     pretrained_emb = self.trans_pretrained(pretrained_emb)
        #     pretrained_emb = pack(pretrained_emb)
        #     inputs += [pretrained_emb]

        # def pad(x):
        #    return pad_packed_sequence(PackedSequence(x, pretrained_emb.batch_sizes), batch_first=True)[0]

        # if self.args['word_emb_dim'] > 0:
        #     word_emb = self.word_emb(word)
        #     word_emb = pack(word_emb)
        # # #     # lemma_emb = self.lemma_emb(lemma)
        # # #     # lemma_emb = pack(lemma_emb)
        # # #     # inputs += [word_emb, lemma_emb]
        #     inputs += [word_emb]
        def pad(x):
            return pad_packed_sequence(PackedSequence(x, phobert_emb.batch_sizes), batch_first=True)[0]

        if self.args['tag_emb_dim'] > 0:
            pos_emb = self.upos_emb(upos)
            pos_emb = pack(pos_emb)
            inputs += [pos_emb]

        # if self.args['char'] and self.args['char_emb_dim'] > 0:
        #     char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
        #     char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
        #     inputs += [char_reps]

        # for x in inputs:
        #     print(x.data.size())
        inputs = torch.cat([x.data for x in inputs], 1)

        inputs = self.worddrop(inputs, self.drop_replacement)
        hidden_out = self.dep_hid(inputs)
        # lstm_inputs = self.drop(lstm_inputs)
        #
        #hidden_out = PackedSequence(hidden_out, inputs[0].batch_sizes)
        #
        # lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens, hx=(self.parserlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.parserlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))

        # lstm_outputs_upos = lstm_outputs.data
        # pos
        # upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs_upos)))
        # upos_pred = self.upos_clf(self.drop(upos_hid))
        #
        # preds = [pad(upos_pred).max(2)[1]]
        #
        # upos = pack(upos).data
        # loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
        ######

        # dep
        # lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        hidden_out = pad(hidden_out)
        unlabeled_scores = self.unlabeled(self.drop(hidden_out), self.drop(hidden_out)).squeeze(3)
        deprel_scores = self.deprel(self.drop(hidden_out), self.drop(hidden_out))

        #goldmask = head.new_zeros(*head.size(), head.size(-1)+1, dtype=torch.uint8)
        #goldmask.scatter_(2, head.unsqueeze(2), 1)

        # dep
        if self.args['linearization'] or self.args['distance']:
            head_offset = torch.arange(word.size(1), device=unlabeled_scores.device).view(1, 1, -1).expand(word.size(
                0), -1, -1) - torch.arange(word.size(1), device=unlabeled_scores.device).view(1, -1, 1).expand(word.size(0), -1, -1)

        if self.args['linearization']:
            #print("USE LINEAR")
            lin_scores = self.linearization(self.drop(hidden_out), self.drop(hidden_out)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()

        if self.args['distance']:
            #print("USE DISTANCE")
            dist_scores = self.distance(self.drop(hidden_out), self.drop(hidden_out)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() - dist_pred)**2/2 + 1)
            unlabeled_scores += dist_kld.detach()
        # print(unlabeled_scores.size())
        diag = torch.eye(unlabeled_scores.size(-1), dtype=torch.bool, device=unlabeled_scores.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))

        preds = []

        if eval == False:
            unlabeled_scores = unlabeled_scores[:, 1:, :]  # exclude attachment for the root symbol
            unlabeled_scores = unlabeled_scores.masked_fill(word_mask.unsqueeze(1), -float('inf'))
            unlabeled_target = head.masked_fill(word_mask[:, 1:], -1)
            loss = self.crit(unlabeled_scores.contiguous().view(-1,
                                                                unlabeled_scores.size(2)), unlabeled_target.view(-1))

            deprel_scores = deprel_scores[:, 1:]  # exclude attachment for the root symbol
            #deprel_scores = deprel_scores.masked_select(goldmask.unsqueeze(3)).view(-1, len(self.vocab['deprel']))
            deprel_scores = torch.gather(deprel_scores, 2, head.unsqueeze(2).unsqueeze(
                3).expand(-1, -1, -1, len(self.vocab['deprel']))).view(-1, len(self.vocab['deprel']))
            deprel_target = deprel.masked_fill(word_mask[:, 1:], -1)
            loss += self.crit(deprel_scores.contiguous(), deprel_target.view(-1))

            if self.args['linearization']:
                #lin_scores = lin_scores[:, 1:].masked_select(goldmask)
                lin_scores = torch.gather(lin_scores[:, 1:], 2, head.unsqueeze(2)).view(-1)
                lin_scores = torch.cat([-lin_scores.unsqueeze(1)/2, lin_scores.unsqueeze(1)/2], 1)
                #lin_target = (head_offset[:, 1:] > 0).long().masked_select(goldmask)
                lin_target = torch.gather((head_offset[:, 1:] > 0).long(), 2, head.unsqueeze(2))
                loss += self.crit(lin_scores.contiguous(), lin_target.view(-1))

            if self.args['distance']:
                #dist_kld = dist_kld[:, 1:].masked_select(goldmask)
                dist_kld = torch.gather(dist_kld[:, 1:], 2, head.unsqueeze(2))
                loss -= dist_kld.sum()

            loss /= wordchars.size(0)  # number of words
        else:
            loss = 0
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())

        return loss, preds
