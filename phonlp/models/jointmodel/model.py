import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from phonlp.models.common.biaffine import DeepBiaffineScorer
from phonlp.models.common.crf import CRFLoss
from phonlp.models.common.dropout import LockedDropout, WordDropout
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, BertPreTrainedModel


class JointModel(BertPreTrainedModel):
    def __init__(self, args, vocab, config):
        super(JointModel, self).__init__(config)

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []
        self.config = config

        # input layers
        self.input_size = 0

        self.phobert = AutoModel.from_pretrained(args["pretrained_lm"], config=self.config)
        self.input_size += self.config.to_dict()["hidden_size"]

        self.drop_replacement_ner = nn.Parameter(
            torch.randn(self.input_size + self.args["tag_emb_dim"])
            / np.sqrt(self.input_size + self.args["tag_emb_dim"])
        )
        self.drop_replacement_dep = nn.Parameter(
            torch.randn(self.input_size + self.args["tag_emb_dim"])
            / np.sqrt(self.input_size + self.args["tag_emb_dim"])
        )
        self.drop_replacement_pos = nn.Parameter(torch.randn(self.input_size) / np.sqrt(self.input_size))

        self.upos_hid = nn.Linear(self.input_size, self.args["deep_biaff_hidden_dim"])
        self.upos_clf = nn.Linear(self.args["deep_biaff_hidden_dim"], len(vocab["upos"]))

        self.upos_emb_matrix_ner = nn.Parameter(
            torch.rand(len(vocab["upos"]), self.args["tag_emb_dim"]), requires_grad=True
        )
        self.upos_emb_matrix_dep = nn.Parameter(
            torch.rand(len(vocab["upos"]), self.args["tag_emb_dim"]), requires_grad=True
        )
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        self.dep_hid = nn.Linear(
            self.input_size + self.args["tag_emb_dim"], self.input_size + self.args["tag_emb_dim"]
        )

        # classifiers
        self.unlabeled = DeepBiaffineScorer(
            self.input_size + self.args["tag_emb_dim"],
            self.input_size + self.args["tag_emb_dim"],
            self.args["deep_biaff_hidden_dim"],
            1,
            pairwise=True,
            dropout=args["dropout"],
        )
        self.deprel = DeepBiaffineScorer(
            self.input_size + self.args["tag_emb_dim"],
            self.input_size + self.args["tag_emb_dim"],
            self.args["deep_biaff_hidden_dim"],
            len(vocab["deprel"]),
            pairwise=True,
            dropout=args["dropout"],
        )
        if args["linearization"]:
            self.linearization = DeepBiaffineScorer(
                self.input_size + self.args["tag_emb_dim"],
                self.input_size + self.args["tag_emb_dim"],
                self.args["deep_biaff_hidden_dim"],
                1,
                pairwise=True,
                dropout=args["dropout"],
            )
        if args["distance"]:
            self.distance = DeepBiaffineScorer(
                self.input_size + self.args["tag_emb_dim"],
                self.input_size + self.args["tag_emb_dim"],
                self.args["deep_biaff_hidden_dim"],
                1,
                pairwise=True,
                dropout=args["dropout"],
            )

        self.ner_tag_clf = nn.Linear(self.input_size + self.args["tag_emb_dim"], len(self.vocab["ner_tag"]))
        self.ner_tag_clf.bias.data.zero_()
        # criterion
        self.crit_ner = CRFLoss(len(self.vocab["ner_tag"]))
        self.crit_dep = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")  # ignore padding
        self.crit_pos = nn.CrossEntropyLoss(ignore_index=1)

        # Dropout
        self.drop_ner = nn.Dropout(args["dropout"])
        self.worddrop_ner = WordDropout(args["word_dropout"])
        self.lockeddrop = LockedDropout(args["word_dropout"])

        self.drop_pos = nn.Dropout(args["dropout"])
        self.worddrop_pos = WordDropout(args["word_dropout"])

        self.drop_dep = nn.Dropout(args["dropout"])
        self.worddrop_dep = WordDropout(args["word_dropout"])

    def pos_forward(self, tokens_phobert, first_subword, sentlens, use_soft_pos=False, upos=None):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        phobert_emb = self.phobert(tokens_phobert)[2][-1]
        phobert_emb = torch.cat(
            [torch.index_select(phobert_emb[i], 0, first_subword[i]).unsqueeze(0) for i in range(phobert_emb.size(0))],
            dim=0,
        )
        if torch.cuda.is_available():
            phobert_emb = phobert_emb.cuda()
        phobert_emb = pack(phobert_emb)
        inputs += [phobert_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, phobert_emb.batch_sizes), batch_first=True)[0]

        inputs_pos = inputs[0].data
        inputs_pos = self.worddrop_pos(inputs_pos, self.drop_replacement_pos)

        upos_hid = F.relu(self.upos_hid(inputs_pos))
        upos_pred = self.upos_clf(self.drop_pos(upos_hid))

        preds = [pad(upos_pred).max(2)[1]]
        if use_soft_pos is False:
            upos = pack(upos).data
            loss = self.crit_pos(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
            return loss, preds
        else:
            return inputs, F.softmax(pad(upos_pred), dim=-1)

    def ner_forward(self, tokens_phobert, first_subword, word_mask, sentlens, tags):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs, pos_dis = self.pos_forward(tokens_phobert, first_subword, sentlens, use_soft_pos=True)
        phobert_emb = inputs[0]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, phobert_emb.batch_sizes), batch_first=True)[0]

        upos_embed_matrix_dup = self.upos_emb_matrix_ner.repeat(pos_dis.size(0), 1, 1)
        pos_emb = torch.matmul(pos_dis, upos_embed_matrix_dup)
        pos_emb = pack(pos_emb)
        inputs += [pos_emb]

        inputs = torch.cat([x.data for x in inputs], 1)
        inputs = self.worddrop_ner(inputs, self.drop_replacement_ner)
        ner_pred = self.ner_tag_clf(inputs)

        logits = pad(F.relu(ner_pred)).contiguous()
        loss, trans = self.crit_ner(logits, word_mask, tags)
        return loss, logits

    def dep_forward(
        self, tokens_phobert, first_subword, word_mask, number_of_words, sentlens, head, deprel, eval=False
    ):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs, pos_dis = self.pos_forward(tokens_phobert, first_subword, sentlens, use_soft_pos=True)
        phobert_emb = inputs[0]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, phobert_emb.batch_sizes), batch_first=True)[0]

        upos_embed_matrix_dup = self.upos_emb_matrix_dep.repeat(pos_dis.size(0), 1, 1)
        pos_emb = torch.matmul(pos_dis, upos_embed_matrix_dup)
        pos_emb = pack(pos_emb)
        inputs += [pos_emb]

        inputs = torch.cat([x.data for x in inputs], 1)
        inputs = self.worddrop_dep(inputs, self.drop_replacement_dep)

        hidden_out = self.dep_hid(inputs)
        hidden_out = pad(hidden_out)

        unlabeled_scores = self.unlabeled(self.drop_dep(hidden_out), self.drop_dep(hidden_out)).squeeze(3)
        deprel_scores = self.deprel(self.drop_dep(hidden_out), self.drop_dep(hidden_out))

        if self.args["linearization"] or self.args["distance"]:
            head_offset = torch.arange(first_subword.size(1), device=unlabeled_scores.device).view(1, 1, -1).expand(
                first_subword.size(0), -1, -1
            ) - torch.arange(first_subword.size(1), device=unlabeled_scores.device).view(1, -1, 1).expand(
                first_subword.size(0), -1, -1
            )

        if self.args["linearization"]:
            lin_scores = self.linearization(self.drop_dep(hidden_out), self.drop_dep(hidden_out)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()

        if self.args["distance"]:
            dist_scores = self.distance(self.drop_dep(hidden_out), self.drop_dep(hidden_out)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() - dist_pred) ** 2 / 2 + 1)
            unlabeled_scores += dist_kld.detach()

        diag = torch.eye(unlabeled_scores.size(-1), dtype=torch.bool, device=unlabeled_scores.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float("inf"))

        preds = []
        if eval is False:
            unlabeled_scores = unlabeled_scores[:, 1:, :]  # exclude attachment for the root symbol
            unlabeled_scores = unlabeled_scores.masked_fill(word_mask.unsqueeze(1), -float("inf"))
            unlabeled_target = head.masked_fill(word_mask[:, 1:], -1)
            loss = self.crit_dep(
                unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1)
            )

            deprel_scores = deprel_scores[:, 1:]  # exclude attachment for the root symbol
            deprel_scores = torch.gather(
                deprel_scores, 2, head.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(self.vocab["deprel"]))
            ).view(-1, len(self.vocab["deprel"]))
            deprel_target = deprel.masked_fill(word_mask[:, 1:], -1)
            loss += self.crit_dep(deprel_scores.contiguous(), deprel_target.view(-1))

            if self.args["linearization"]:
                # lin_scores = lin_scores[:, 1:].masked_select(goldmask)
                lin_scores = torch.gather(lin_scores[:, 1:], 2, head.unsqueeze(2)).view(-1)
                lin_scores = torch.cat([-lin_scores.unsqueeze(1) / 2, lin_scores.unsqueeze(1) / 2], 1)
                lin_target = torch.gather((head_offset[:, 1:] > 0).long(), 2, head.unsqueeze(2))
                loss += self.crit_dep(lin_scores.contiguous(), lin_target.view(-1))

            if self.args["distance"]:
                dist_kld = torch.gather(dist_kld[:, 1:], 2, head.unsqueeze(2))
                loss -= dist_kld.sum()

            loss /= number_of_words  # number of words
        else:
            loss = 0
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return loss, preds

    def forward(
        self,
        dep_tokens_phobert,
        dep_first_subword,
        dep_word_mask,
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
        ner_word_mask,
        ner_sentlens,
        ner_tag,
        lambda_pos=1.0,
        lambda_ner=1.0,
        lambda_dep=1.0,
    ):
        loss_pos, preds_pos = self.pos_forward(pos_tokens_phobert, pos_first_subword, pos_sentlens, False, pos_upos)
        loss_ner, preds_ner = self.ner_forward(
            ner_tokens_phobert, ner_first_subword, ner_word_mask, ner_sentlens, ner_tag
        )
        loss_dep, preds_dep = self.dep_forward(
            dep_tokens_phobert,
            dep_first_subword,
            dep_word_mask,
            dep_number_of_words,
            dep_sentlens,
            dep_head,
            dep_deprel,
        )

        loss = lambda_pos * loss_pos + lambda_ner * loss_ner + lambda_dep * loss_dep
        preds = preds_pos + [preds_ner] + preds_dep
        return loss_pos, loss_ner, loss_dep, loss, preds
