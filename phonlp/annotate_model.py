# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from phonlp.models.common import utils as util
from phonlp.models.common.biaffine import DeepBiaffineScorer
from phonlp.models.common.chuliu_edmonds import chuliu_edmonds_one_root
from phonlp.models.common.crf import CRFLoss, viterbi_decode
from phonlp.models.common.data import get_long_tensor, sort_all
from phonlp.models.common.dropout import LockedDropout, WordDropout
from phonlp.models.common.vocab import PAD_ID, ROOT_ID
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from transformers import AutoModel, BertPreTrainedModel


class JointModel(BertPreTrainedModel):
    def __init__(self, args, vocab, config, tokenizer):
        super(JointModel, self).__init__(config)

        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []
        self.config = config
        self.tokenizer = tokenizer

        # input layers
        self.input_size = 0

        self.phobert = AutoModel.from_config(self.config)
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
        self.drop_ner = nn.Dropout(args["dropout"])
        self.worddrop_ner = WordDropout(args["word_dropout"])
        self.lockeddrop = LockedDropout(args["word_dropout"])

        self.drop_pos = nn.Dropout(args["dropout"])
        self.worddrop_pos = WordDropout(args["word_dropout"])

        self.drop_dep = nn.Dropout(args["dropout"])
        self.worddrop_dep = WordDropout(args["word_dropout"])

    def tagger_forward(self, tokens_phobert, words_phobert, sentlens):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []

        phobert_emb = self.phobert(tokens_phobert)[2][-1]
        phobert_emb = torch.cat(
            [torch.index_select(phobert_emb[i], 0, words_phobert[i]).unsqueeze(0) for i in range(phobert_emb.size(0))],
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

        preds_pos = [pad(upos_pred).max(2)[1]]

        pos_dis = F.softmax(pad(upos_pred), dim=-1)
        upos_embed_matrix_dup = self.upos_emb_matrix_ner.repeat(pos_dis.size(0), 1, 1)
        pos_emb = torch.matmul(pos_dis, upos_embed_matrix_dup)
        pos_emb = pack(pos_emb)
        inputs += [pos_emb]

        inputs = torch.cat([x.data for x in inputs], 1)
        inputs = self.worddrop_ner(inputs, self.drop_replacement_ner)

        ner_pred = self.ner_tag_clf(inputs)
        logits = pad(F.relu(ner_pred)).contiguous()

        return preds_pos, logits

    def dep_forward(self, tokens_phobert, first_subword, sentlens):
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
        pos_dis = F.softmax(pad(upos_pred), dim=-1)
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
        preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
        preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return preds

    def annotate(self, text=None, input_file=None, output_file=None, batch_size=1, output_type=""):
        if text is not None:
            data = [text.split(" ")]
        else:
            f = open(input_file)
            data = []
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    data.append(line.split(" "))
            f.close()
            print("The number of sentences: ", len(data))
        data_tagger = self.process_data_tagger(batch_text=data)
        data_parser = self.process_data_parser(batch_text=data)
        data_parser = self.chunk_batches(data_parser, batch_size)
        data_tagger = self.chunk_batches(data_tagger, batch_size)
        test_preds_pos = []
        test_preds_dep = []
        test_preds_ner = []
        for i in tqdm(range(len(data_tagger))):
            tokens_phobert, first_subword, words_mask, number_of_words, orig_idx, sentlens = self.get_batch(
                i, data_tagger
            )
            tokens_phobert1, first_subword1, words_mask1, number_of_words1, orig_idx1, sentlens1 = self.get_batch(
                i, data_parser
            )
            if torch.cuda.is_available():
                tokens_phobert, first_subword, words_mask = (
                    tokens_phobert.cuda(),
                    first_subword.cuda(),
                    words_mask.cuda(),
                )
                tokens_phobert1, first_subword1, words_mask1 = (
                    tokens_phobert1.cuda(),
                    first_subword1.cuda(),
                    words_mask1.cuda(),
                )

            preds_dep = self.dep_forward(tokens_phobert1, first_subword1, sentlens1)
            preds_pos, logits = self.tagger_forward(tokens_phobert, first_subword, sentlens)
            batch_size = tokens_phobert.size(0)
            # DEP
            head_seqs = [
                chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds_dep[0], sentlens1)
            ]  # remove attachment for the root
            deprel_seqs = [
                self.vocab["deprel"].unmap([preds_dep[1][i][j + 1][h] for j, h in enumerate(hs)])
                for i, hs in enumerate(head_seqs)
            ]
            pred_tokens = [
                [[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens1[i] - 1)] for i in range(batch_size)
            ]
            pred_tokens_dep = util.unsort(pred_tokens, orig_idx1)

            # POS
            upos_seqs = [self.vocab["upos"].unmap(sent) for sent in preds_pos[0]]
            pred_tokens_pos = [
                [[upos_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)
            ]  # , xpos_seqs[i][j], feats_seqs[i][j]
            pred_tokens_pos = util.unsort(pred_tokens_pos, orig_idx)

            trans = self.crit_ner._transitions.data.cpu().numpy()
            scores = logits.data.cpu().numpy()
            bs = logits.size(0)
            tag_seqs = []
            for i in range(bs):
                tags, _ = viterbi_decode(scores[i, : sentlens[i]], trans)
                tags = self.vocab["ner_tag"].unmap(tags)
                tag_seqs += [tags]
            tag_seqs = util.unsort(tag_seqs, orig_idx)
            test_preds_ner += tag_seqs
            test_preds_dep += pred_tokens_dep
            test_preds_pos += pred_tokens_pos
        test_preds_dep = util.unsort(test_preds_dep, self.data_orig_idx)
        test_preds_pos = util.unsort(test_preds_pos, self.data_orig_idx)
        test_preds_ner = util.unsort(test_preds_ner, self.data_orig_idx)
        if text is not None:
            return (data, test_preds_pos, test_preds_ner, test_preds_dep)
        else:
            f = open(output_file, "w")
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if output_type == "conll":
                        f.write(
                            str(j + 1)
                            + "\t"
                            + data[i][j]
                            + "\t"
                            + "_"
                            + "\t"
                            + "_"
                            + "\t"
                            + test_preds_pos[i][j][0]
                            + "\t"
                            + "_"
                            + "\t"
                            + test_preds_dep[i][j][0]
                            + "\t"
                            + test_preds_dep[i][j][1]
                            + "\t"
                            + "_"
                            + "\t"
                            + test_preds_ner[i][j]
                            + "\n"
                        )
                    else:
                        f.write(
                            str(j + 1)
                            + "\t"
                            + data[i][j]
                            + "\t"
                            + test_preds_pos[i][j][0]
                            + "\t"
                            + test_preds_ner[i][j]
                            + "\t"
                            + test_preds_dep[i][j][0]
                            + "\t"
                            + test_preds_dep[i][j][1]
                            + "\n"
                        )
                f.write("\n")
            f.close()

    def print_out(self, output, output_type=""):
        data, test_preds_pos, test_preds_ner, test_preds_dep = output
        for i in range(len(data)):
            for j in range(len(data[i])):
                if output_type == "conll":
                    print(
                        str(j + 1)
                        + "\t"
                        + data[i][j]
                        + "\t"
                        + "_"
                        + "\t"
                        + "_"
                        + "\t"
                        + test_preds_pos[i][j][0]
                        + "\t"
                        + "_"
                        + "\t"
                        + test_preds_dep[i][j][0]
                        + "\t"
                        + test_preds_dep[i][j][1]
                        + "\t"
                        + "_"
                        + "\t"
                        + test_preds_ner[i][j]
                    )
                else:
                    print(
                        str(j + 1)
                        + "\t"
                        + data[i][j]
                        + "\t"
                        + test_preds_pos[i][j][0]
                        + "\t"
                        + test_preds_ner[i][j]
                        + "\t"
                        + test_preds_dep[i][j][0]
                        + "\t"
                        + test_preds_dep[i][j][1]
                    )

    def process_data_tagger(self, batch_text):
        cls_id = 0
        sep_id = 2
        processed = []
        for sent in batch_text:
            input_ids = [cls_id]
            firstSWindices = [len(input_ids)]
            for w in sent:
                word_token = self.tokenizer.encode(w)
                input_ids += word_token[1 : (len(word_token) - 1)]
                firstSWindices.append(len(input_ids))
            firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
            input_ids.append(sep_id)
            processed_sent = [input_ids]
            processed_sent += [firstSWindices]
            processed_sent += [self.vocab["word"].map([w for w in sent])]
            processed_sent += [[self.vocab["char"].map([x for x in w]) for w in sent]]
            processed.append(processed_sent)
        return processed

    def process_data_parser(self, batch_text):
        cls_id = 0
        sep_id = 2
        processed = []
        for sent in batch_text:
            input_ids = [cls_id]
            firstSWindices = [len(input_ids)]
            root_token = self.tokenizer.encode("[ROOT]")
            input_ids += root_token[1 : (len(root_token) - 1)]
            firstSWindices.append(len(input_ids))
            for w in sent:
                word_token = self.tokenizer.encode(w)
                input_ids += word_token[1 : (len(word_token) - 1)]
                firstSWindices.append(len(input_ids))
            firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
            input_ids.append(sep_id)

            processed_sent = [input_ids]
            processed_sent += [firstSWindices]
            processed_sent += [[ROOT_ID] + self.vocab["word"].map([w for w in sent])]
            processed_sent += [[[ROOT_ID]] + [self.vocab["char"].map([x for x in w]) for w in sent]]
            processed.append(processed_sent)
        return processed

    def chunk_batches(self, data, batch_size):
        res = []
        (data,), self.data_orig_idx = sort_all([data], [len(x[2]) for x in data])
        current = []
        for x in data:
            if len(current) >= batch_size:
                res.append(current)
                current = []
            current.append(x)

        if len(current) > 0:
            res.append(current)
        return res

    def get_batch(self, key, data_chunk):
        batch = data_chunk[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 4
        # print(batch)
        lens = [len(x) for x in batch[2]]
        batch, orig_idx = sort_all(batch, lens)

        batch_words = [w for sent in batch[3] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]  # [word1,...], word1 = list of tokens
        word_lens = [len(x) for x in batch_words]
        wordchars = get_long_tensor(batch_words, len(word_lens))
        number_of_words = wordchars.size(0)
        words = batch[2]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)

        # convert to tensors
        tokens_phobert = batch[0]
        tokens_phobert = get_long_tensor(tokens_phobert, batch_size, pad_id=1)
        first_subword = batch[1]
        first_subword = get_long_tensor(first_subword, batch_size)
        sentlens = [len(x) for x in batch[1]]
        data = (tokens_phobert, first_subword, words_mask, number_of_words, orig_idx, sentlens)
        return data
