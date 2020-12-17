# -*- coding: utf-8 -*-
# import sys
#
# sys.path.append('./')
# sys.path.append('./phonlp')
from phonlp.models.common import utils as util
from tqdm import tqdm
from phonlp.models.common.chuliu_edmonds import chuliu_edmonds_one_root
# from phonlp.models.jointmodel3task.model import *
import torch
from phonlp.models.common.crf import viterbi_decode
from phonlp.models.common.data import get_long_tensor, sort_all
from phonlp.models.common.vocab import PAD_ID, ROOT_ID
from phonlp.model_eval import JointModel
from phonlp.models.ner.vocab import MultiVocab
from transformers import AutoConfig, AutoTokenizer
import gdown
import os


def download(path_save_model, language='vi'):
    url = "https://drive.google.com/uc?id=1ZFfyppGc4QKdeGve1kvpj44GTlM9Rl-H"
    # rootdir = os.getenv('HOME')
    # if os.path.exists(rootdir + '/.cache'):
    #     if os.path.exists(rootdir + '/.cache/phonlp'):
    #         if os.path.exists(rootdir + '/.cache/phonlp/phonlp_model.pt'):
    #             pass
    #         else:
    #             gdown.download(url, rootdir + '/.cache/phonlp/phonlp_model.pt')
    #     else:
    #         os.mkdir(rootdir + '/.cache/phonlp')
    #         gdown.download(url, rootdir + '/.cache/phonlp/phonlp_model.pt')
    # else:
    #     os.mkdir(rootdir + '/.cache')
    #     os.mkdir(rootdir + '/.cache/phonlp')
    #     gdown.download(url, rootdir + '/.cache/phonlp/phonlp_model.pt')
    if language == 'vi':
        gdown.download(url, path_save_model)
    else:
        raise ValueError('Unrecognized language. Please choose language.')


def load_model(path_save_model):
    #model_file = "/home/vinai/Documents/PhoToolkit/phonlp/models/save_model/VnDTv1.1_jointmodel.pt"
    if path_save_model[len(path_save_model) - 1] == '/':
        model_file = path_save_model + "VnDTv1.1_jointmodel.pt"
    else:
        model_file = path_save_model + "/VnDTv1.1_jointmodel.pt"
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    config_phobert = AutoConfig.from_pretrained('vinai/phobert-base', output_hidden_states=True)
    print("Loading model from: {}".format(model_file))
    use_cuda = False  # args['cuda'] and not args['cpu']
    # trainer = TrainerJoint(model_file=model_file, use_cuda=use_cuda, config_phobert=config_phobert)
    # trainer.model.to(torch.device('cpu'))
    checkpoint = torch.load(model_file, lambda storage, loc: storage)
    args = checkpoint['config']
    vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
    # load model
    model = JointModel(args, vocab, config=config_phobert)
    model.load_state_dict(checkpoint['model'], strict=False)
    if torch.cuda.is_available() == False:
        model.to(torch.device('cpu'))
    else:
        model.to(torch.device('cuda'))
    return (model, vocab, tokenizer)


def annotate(model, text=None, input_file=None, output_file=None, type='corpus'):
    model, vocab, tokenizer = model
    if type == 'corpus':
        f = open(input_file)
        doc = []
        for line in f:
            line = line.strip()
            if len(line) != 0:
                doc.append(line)
        f.close()
        print("The number of sentences: ", len(doc))
        f = open(output_file, 'w')
        for i in tqdm(range(len(doc))):
            pred_tokens, pred_tokens_pos, tag_seqs = annotate_sentence(doc[i], model, tokenizer, vocab)
            text_list = doc[i].split(' ')
            for j in range(len(text_list)):
                f.write(str(j + 1) + '\t' + text_list[j] + '\t' + '_' + '\t' + '_' + '\t' + pred_tokens_pos[0][j][
                    0] + '\t' + '_' + '\t' + pred_tokens[0][j][0] + '\t' + pred_tokens[0][j][1] + '\t' + '_' + '\t' +
                    tag_seqs[0][j] + '\n')
            f.write('\n')
        f.close()
    elif type == 'sentence':
        pred_tokens, pred_tokens_pos, tag_seqs = annotate_sentence(text, model, tokenizer, vocab)
        text_list = text.split(' ')
        for j in range(len(text_list)):
            print(str(j + 1) + '\t' + text_list[j] + '\t' + '_' + '\t' + '_' + '\t' + pred_tokens_pos[0][j][
                0] + '\t' + '_' + '\t' + pred_tokens[0][j][0] + '\t' + pred_tokens[0][j][1] + '\t' + '_' + '\t' +
                tag_seqs[0][j] + '\n')


def annotate_sentence(text, model, tokenizer, vocab):
    # loaded_args, vocab = trainer.args, trainer.vocab
    tokens_phobert, words_phobert, words, words_mask, wordchars, wordchars_mask, orig_idx, word_orig_idx, sentlens, word_lens = get_batch(
        process_data_tagger(text, vocab, tokenizer))
    tokens_phobert1, words_phobert1, words1, words_mask1, wordchars1, wordchars_mask1, orig_idx1, word_orig_idx1, sentlens1, word_lens1 = get_batch(
        process_data_parser(text, vocab, tokenizer))

    if torch.cuda.is_available():
        tokens_phobert, words_phobert, words, words_mask, wordchars, wordchars_mask = tokens_phobert.cuda(
        ), words_phobert.cuda(), words.cuda(), words_mask.cuda(), wordchars.cuda(), wordchars_mask.cuda()
        tokens_phobert1, words_phobert1, words1, words_mask1, wordchars1, wordchars_mask1 = tokens_phobert1.cuda(
        ), words_phobert1.cuda(), words1.cuda(), words_mask1.cuda(), wordchars1.cuda(), wordchars_mask1.cuda()

    model.eval()
    # dep
    batch_size = 1
    loss_dep, preds_dep = model.dep_forward(tokens_phobert1, words_phobert1, words1, words_mask1, wordchars1,
                                            sentlens1, eval=True)
    head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                 zip(preds_dep[0], sentlens1)]  # remove attachment for the root
    deprel_seqs = [vocab['deprel'].unmap([preds_dep[1][i][j + 1][h] for j, h in enumerate(hs)]) for i, hs in
                   enumerate(head_seqs)]
    pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens1[i] - 1)] for i in
                   range(batch_size)]
    pred_tokens = util.unsort(pred_tokens, orig_idx1)
    # print(pred_tokens)
    # pos
    preds_pos, _ = model.pos_forward(tokens_phobert, words_phobert, sentlens, eval=True)
    # print(preds_pos)
    upos_seqs = [vocab['upos'].unmap(sent) for sent in preds_pos[0]]
    pred_tokens_pos = [[[upos_seqs[i][j]] for j in range(sentlens[i])] for i in
                       range(batch_size)]  # , xpos_seqs[i][j], feats_seqs[i][j]
    pred_tokens_pos = util.unsort(pred_tokens_pos, orig_idx)
    # print(pred_tokens_pos)
    # ner
    logits = model.ner_forward(tokens_phobert, words_phobert, words_mask, sentlens, eval=True)
    trans = model.crit_ner._transitions.data.cpu().numpy()
    scores = logits.data.cpu().numpy()
    bs = logits.size(0)
    tag_seqs = []
    for i in range(bs):
        tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
        tags = vocab['ner_tag'].unmap(tags)
        tag_seqs += [tags]
    tag_seqs = util.unsort(tag_seqs, orig_idx)
    # print(tag_seqs)
    # print(pred_tokens)
    # print(pred_tokens_pos)
    # print(tag_seqs)
    # print
    ##############################
    # text_list = text.split(' ')
    # for i in range(len(text_list)):
    #     print(str(i + 1) + '\t' + text_list[i] + '\t' + '_' + '\t' + '_' + '\t' + pred_tokens_pos[0][i][
    #         0] + '\t' + '_' + '\t' + pred_tokens[0][i][0] + '\t' + pred_tokens[0][i][1] + '\t' + '_' + '\t' +
    #           tag_seqs[0][i] + '\n')

    return pred_tokens, pred_tokens_pos, tag_seqs


def process_data_tagger(text, vocab, tokenizer):
    pad_id = 1
    cls_id = 0
    sep_id = 2
    sent = text.split(' ')
    processed = []
    input_ids = [cls_id]
    firstSWindices = [len(input_ids)]
    for w in sent:
        word_token = tokenizer.encode(w)
        input_ids += word_token[1:(len(word_token) - 1)]
        firstSWindices.append(len(input_ids))
    firstSWindices = firstSWindices[:(len(firstSWindices) - 1)]
    input_ids.append(sep_id)
    # if len(input_ids) > max_seq_length:
    #     input_ids = input_ids[:max_seq_length]
    #     # input_ids[-1] = eos_id
    # else:
    #     input_ids = input_ids + [pad_id, ] * (max_seq_length - len(input_ids))
    processed_sent = [input_ids]
    processed_sent += [firstSWindices]
    processed_sent += [vocab['word'].map([w for w in sent])]
    processed_sent += [[vocab['char'].map([x for x in w]) for w in sent]]
    processed.append(processed_sent)

    return processed


def process_data_parser(text, vocab, tokenizer):
    pad_id = 1
    cls_id = 0
    sep_id = 2
    sent = text.split(' ')
    processed = []
    input_ids = [cls_id]
    firstSWindices = [len(input_ids)]
    root_token = tokenizer.encode("[ROOT]")
    input_ids += root_token[1:(len(root_token) - 1)]
    firstSWindices.append(len(input_ids))
    for w in sent:
        word_token = tokenizer.encode(w)
        input_ids += word_token[1:(len(word_token) - 1)]
        firstSWindices.append(len(input_ids))
    firstSWindices = firstSWindices[:(len(firstSWindices) - 1)]
    input_ids.append(sep_id)
    # if len(input_ids) > max_seq_length:
    #     input_ids = input_ids[:max_seq_length]
    #     # input_ids[-1] = eos_id
    # else:
    #     input_ids = input_ids + [pad_id, ] * (max_seq_length - len(input_ids))
    processed_sent = [input_ids]
    processed_sent += [firstSWindices]
    processed_sent += [[ROOT_ID] + vocab['word'].map([w for w in sent])]
    processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w]) for w in sent]]
    processed.append(processed_sent)

    return processed


def get_batch(processed):
    batch = processed[0]
    batch_size = 1
    #batch = list(zip(*batch))
    # print(batch)
    for i in range(len(batch)):
        batch[i] = [batch[i]]
    # print(batch)
    lens = [len(x) for x in batch[2]]
    batch, orig_idx = sort_all(batch, lens)

    # sort words by lens for easy char-RNN operations
    batch_words = [w for sent in batch[3] for w in sent]
    word_lens = [len(x) for x in batch_words]
    batch_words, word_orig_idx = sort_all([batch_words], word_lens)
    batch_words = batch_words[0]
    word_lens = [len(x) for x in batch_words]

    # convert to tensors
    tokens_phobert = batch[0]
    tokens_phobert = get_long_tensor(tokens_phobert, batch_size, pad_id=1)
    words_phobert = batch[1]
    words_phobert = get_long_tensor(words_phobert, batch_size)
    words = batch[2]
    words = get_long_tensor(words, batch_size)
    words_mask = torch.eq(words, PAD_ID)
    wordchars = get_long_tensor(batch_words, len(word_lens))
    wordchars_mask = torch.eq(wordchars, PAD_ID)
    sentlens = [len(x) for x in batch[2]]
    # print(sentlens)
    return tokens_phobert, words_phobert, words, words_mask, wordchars, wordchars_mask, orig_idx, word_orig_idx, sentlens, word_lens


if __name__ == '__main__':
    download('vi')
#     text = "Tôi tên là Thế_Linh ."
#     annotate(text=text, type='sentence')#input_file='input.txt', output_file='output.txt')
