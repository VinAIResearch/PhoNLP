# -*- coding: utf-8 -*-
"""
Entry point for training and evaluating a POS/morphological features tagger.

This tagger uses highway BiLSTM layers with character and word-level representations, and biaffine classifiers
to produce consistant POS and UFeats predictions.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

from tqdm import tqdm
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import *
from transformers.modeling_utils import *
from utils.conll import CoNLL
from models.common import utils as util
from models.pos import scorer
from models.pos.trainer import TrainerPOS
from models.pos.data import DataLoaderPOS
import sys
import os
import shutil
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
from torch import nn, optim
sys.path.append('../')
#from models.common.pretrain import Pretrain
#from models.common.doc import *
#from PhoToolkit.models import _training_logging
#from fairseq.data.encoders.fastbpe import fastBPE
#from fairseq.data import Dictionary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default="/home/ubuntu/linhnt140/data/POS_data/POS_data/VLSP2013_POS_train.txt", help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str,
                        default="/home/ubuntu/linhnt140/data/POS_data/POS_data/VLSP2013_POS_dev.txt", help='Input file for data loader.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, default='VnDTv1.1', help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_true', help="Turn off pretrained embeddings.")
    parser.add_argument('--share_hid', action='store_true',
                        help="Share hidden representations for UPOS, XPOS and UFeats.")
    parser.set_defaults(share_hid=False)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adamax', help='sgd, adagrad, adamw, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--fix_eval_interval', dest='adapt_eval_interval', action='store_false',
                        help="Use fixed evaluation interval for all treebanks, otherwise by default the interval will be increased for larger treebanks.")
    parser.add_argument('--max_steps_before_stop', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models_newversion/pos',
                        help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    # phobert
    parser.add_argument('--use_phobert', type=bool, default=True)
    parser.add_argument('--dict_path', type=str, default="/home/ubuntu/linhnt140/PhoBERT_base_transformers/dict.txt")
    parser.add_argument('--config_path', type=str,
                        default="/home/ubuntu/linhnt140/PhoBERT_base_transformers/config.json")
    parser.add_argument('--phobert_model', type=str, default='vinai/phobert-base')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--bpe-codes', default="/home/ubuntu/linhnt140/PhoBERT_base_transformers/bpe.codes", type=str,
                        help='path to fastBPE BPE')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    print("Running tagger in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)


def train(args):
    util.ensure_dir(args['save_dir'])
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_tagger.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors if needed
    # pretrain = None
    # if args['pretrain']:
    #     #vec_file = util.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    #     vec_file = args['wordvec_dir'] + "word2vec_vi_words_300dims.txt"
    #     pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    #     pretrain = Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])

    tokenizer = AutoTokenizer.from_pretrained(args['phobert_model'])
    config_phobert = AutoConfig.from_pretrained(args['phobert_model'], output_hidden_states=True)

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    # train_doc = Document(CoNLL.conll2dict(input_file=args['train_file']))
    train_batch = DataLoaderPOS(args['train_file'], args['batch_size'], args, vocab=None,
                                evaluation=False, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    print("Number of train example: ", len(train_batch))
    vocab = train_batch.vocab
    print(vocab['upos']._unit2id)
    # dev_doc = Document(CoNLL.conll2dict(input_file=args['eval_file']))
    dev_batch = DataLoaderPOS(args['eval_file'], args['batch_size'], args, vocab=vocab, evaluation=True,
                              sort_during_eval=True, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])

    # train_batch_test = DataLoader(train_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True, vocab_phobert=vocab_phobert, bpe=bpe, max_seq_length=args['max_sequence_length'])

    # pred and gold path
    # system_pred_file = args['output_file']
    # gold_file = args['gold_file']

    # system_pred_file_train = args['output_file_train']
    # gold_file_train = args['train_file']

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    print("Training tagger...")
    trainer = TrainerPOS(args, vocab, None, config_phobert, args['cuda'])
    ###
    tsfm = trainer.model.phobert
    tq = tqdm(range(args['num_epoch'] + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = True
    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    # if args['adapt_eval_interval']:
    #     args['eval_interval'] = util.get_adaptive_eval_interval(dev_batch.num_examples, 2000, args['eval_interval'])
    #     print("Evaluating the model every {} steps...".format(args['eval_interval']))

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    # while True:
    #     do_break = False

    # Creating optimizer and lr schedulers
    param_optimizer = list(trainer.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(args['num_epoch'] * len(train_batch) * args['accumulation_steps'])
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['lr'],
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                num_training_steps=num_train_optimization_steps)
    scheduler0 = get_constant_schedule(optimizer)

    for epoch in range(args['num_epoch']):
        optimizer.zero_grad()
        print(" EPOCH  : ", epoch)
        pbar = tqdm(enumerate(train_batch), total=len(train_batch), leave=False)
        for i, batch in pbar:
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False)  # update step
            train_loss += loss
            if i % args['accumulation_steps'] == 0:
                # torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), args['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            # if i % args['eval_interval']:
            #     print("Evaluating on dev set...")
            #     dev_preds = []
            #     for batch in dev_batch:
            #         preds = trainer.predict(batch)
            #         dev_preds += preds
            #     dev_preds = util.unsort(dev_preds, dev_batch.data_orig_idx_pos)
            #
            #     # dev_batch.doc.set([UPOS, XPOS, FEATS], [y for x in dev_preds for y in x])
            #     # CoNLL.dict2conll(dev_batch.doc.to_dict(), system_pred_file)
            #     # p, r, dev_score = scorer.score(system_pred_file, gold_file)
            #
            #     dev_score = scorer.score_acc(dev_preds, dev_batch.upos)
            #     print("Epoch: {}: dev_score = {:.4f}".format(epoch, dev_score))
            # # if global_step % args['log_step'] == 0:
            #     duration = time.time() - start_time
            #     print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
            #             max_steps, loss, duration, current_lr))

            # if global_step % args['eval_interval'] == 0:
                # eval on dev
        print("Evaluating on dev set...")
        dev_preds = []
        for batch in dev_batch:
            preds = trainer.predict(batch)
            dev_preds += preds
        dev_preds = util.unsort(dev_preds, dev_batch.data_orig_idx_pos)

        # dev_batch.doc.set([UPOS, XPOS, FEATS], [y for x in dev_preds for y in x])
        # CoNLL.dict2conll(dev_batch.doc.to_dict(), system_pred_file)
        # p, r, dev_score = scorer.score(system_pred_file, gold_file)

        dev_score = scorer.score_acc(dev_preds, dev_batch.upos)

        train_loss = train_loss / len(train_batch)  # avg loss per batch
        print("Epoch: {}, step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(epoch, global_step, train_loss, dev_score))

        # print("Evaluating on train set...")
        # train_preds = []
        # for batch in train_batch_test:
        #     preds = trainer.predict(batch)
        #     train_preds += preds
        # train_preds = util.unsort(train_preds, train_batch_test.data_orig_idx)
        # # train_batch_test.doc.set([UPOS, XPOS, FEATS], [y for x in train_preds for y in x])
        # # CoNLL.dict2conll(train_batch_test.doc.to_dict(), system_pred_file_train)
        # # p, r, train_score = scorer.score(system_pred_file_train, gold_file_train)
        #
        # train_score = scorer.score_acc(train_preds, train_batch_test.upos)
        #
        # # train_loss = train_loss / len(train_batch)  # avg loss per batch
        # print("Epoch: {}, step {}: train_loss = {:.6f}, train_score = {:.4f}".format(epoch, global_step, train_loss,
        #                                                                            train_score))
        train_loss = 0
        # save best model
        if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
            last_best_step = global_step
            trainer.save(model_file)
            print("new best model saved.")
            best_dev_preds = dev_preds

        dev_score_history += [dev_score]
        print("")

        # if global_step - last_best_step >= args['max_steps_before_stop']:
        #     if not using_amsgrad:
        #         print("Switching to AMSGrad")
        #         last_best_step = global_step
        #         using_amsgrad = True
        #         trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-6)
        #     # else:
        #     do_break = True
        #     break

        # if global_step >= args['max_steps']:
        #     do_break = True
        #     break

        # if do_break: break

        train_batch.reshuffle()

    print("Training ended with {} steps.".format(global_step))

    best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
    print("Best dev F1 = {:.2f}, at epoch = {}".format(best_f, best_eval))


def evaluate(args):
    # file paths
    # system_pred_file = args['output_file']
    # gold_file = args['gold_file']
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_tagger.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrain; note that we allow the pretrain_file to be non-existent
    # pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    # pretrain = Pretrain(pretrain_file)
    # pretrain = None
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'])
    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    config_phobert = AutoConfig.from_pretrained(args["phobert_model"], output_hidden_states=True)
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args["phobert_model"])
    print("Loading model from: {}".format(model_file))
    use_cuda = args['cuda'] and not args['cpu']
    trainer = TrainerPOS(model_file=model_file, use_cuda=use_cuda, config_phobert=config_phobert)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    # for k in args:
    #     if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
    #         loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    #doc = Document(CoNLL.conll2dict(input_file=args['eval_file']))
    batch = DataLoaderPOS(args['eval_file'], args['batch_size'], args, vocab=vocab, evaluation=True,
                          sort_during_eval=True, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    if len(batch) > 0:
        print("Start evaluation...")
        preds = []
        for i, b in enumerate(batch):
            preds += trainer.predict(b)
    else:
        # skip eval if dev data does not exist
        preds = []
    preds = util.unsort(preds, batch.data_orig_idx_pos)

    # write to file and score
    score = scorer.score_acc(preds, batch.upos)
    # batch.doc.set([UPOS], [y[0] for x in preds for y in x])
    # CoNLL.dict2conll(batch.doc.to_dict(), system_pred_file)

    # #
    # if gold_file is not None:
    #     _, _, score_file = scorer.score(system_pred_file, gold_file)

    print("Tagger score:")
    print("{} {:.2f}".format(args['shorthand'], score*100))


if __name__ == '__main__':
    main()
