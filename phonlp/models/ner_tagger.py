# -*- coding: utf-8 -*-
"""
Entry point for training and evaluating an NER tagger.

This tagger uses BiLSTM layers with character and word-level representations, and a CRF decoding layer
to produce NER predictions.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import *
from transformers.modeling_utils import *
from models.common import utils as util
from models.ner import scorer
from models.ner.trainer import TrainerNER
from models.ner.data import DataLoader
import sys
import os
import time
from datetime import datetime
import argparse
import logging
import numpy as np
import random
import json
import torch
from torch import nn, optim
sys.path.append('../')
# from PhoToolkit.models.common.pretrain import Pretrain
# from PhoToolkit.utils.conll import CoNLL
# from PhoToolkit.models.common.doc import *
# from PhoToolkit.models import _training_logging
# from fairseq.data.encoders.fastbpe import fastBPE
# from fairseq.data import Dictionary

# logger = logging.getLogger('PhoToolkit')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default="/home/ubuntu/linhnt140/data/NER_data/train.txt", help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default="/home/ubuntu/linhnt140/data/NER_data/dev.txt",
                        help='Input file for data loader.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, default="NER", help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--transformed_dim', type=int, default=300)

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=100000)
    parser.add_argument('--word_dropout', type=float, default=0.3)
    parser.add_argument('--locked_dropout', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Word recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Character recurrent dropout")
    parser.add_argument('--char_dropout', type=float, default=0, help="Character-level language model dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--charlm', action='store_true',
                        help="Turn on contextualized char embedding using character-level language model.")
    parser.add_argument('--charlm_save_dir', type=str, default='saved_models/charlm',
                        help="Root dir for pretrained character-level language model.")
    parser.add_argument('--charlm_shorthand', type=str, default=None,
                        help="Shorthand for character-level language model training corpus.")
    parser.add_argument('--char_lowercase', dest='char_lowercase', action='store_true',
                        help="Use lowercased characters in charater model.")
    parser.add_argument('--no_lowercase', dest='lowercase', action='store_false', help="Use cased word vectors.")
    parser.add_argument('--no_emb_finetune', dest='emb_finetune', action='store_false',
                        help="Turn off finetuning of the embedding matrix.")
    #parser.add_argument('--no_input_transform', dest='input_transform', action='store_false', help="Do not use input transformation layer before tagger lstm.")
    parser.add_argument('--scheme', type=str, default='bioes', help="The tagging scheme to use: bio or bioes.")
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adamax', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate to stop training.')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for SGD.')
    parser.add_argument('--lr_decay', type=float, default=0.5, help="LR decay rate.")
    parser.add_argument('--patience', type=int, default=2, help="Patience for LR decay.")
    parser.add_argument('--num_epoch', type=int, default=40)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models_newversion/ner',
                        help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

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
        else '{}/{}_nertagger.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors
    # if len(args['wordvec_file']) == 0:
    #     vec_file = util.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    # else:
    #     vec_file = args['wordvec_file']
    # # do not save pretrained embeddings individually
    # pretrain = Pretrain(None, vec_file, args['pretrain_max_vocab'], save_to_file=False)
    # pretrain = None

    # if args['charlm']:
    #     if args['charlm_shorthand'] is None:
    #         print("CharLM Shorthand is required for loading pretrained CharLM model...")
    #         sys.exit(0)
    #     print('Use pretrained contextualized char embedding')
    #     args['charlm_forward_file'] = '{}/{}_forward_charlm.pt'.format(args['charlm_save_dir'], args['charlm_shorthand'])
    #     args['charlm_backward_file'] = '{}/{}_backward_charlm.pt'.format(args['charlm_save_dir'], args['charlm_shorthand'])

    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args['phobert_model'])
    config_phobert = AutoConfig.from_pretrained(args['phobert_model'], output_hidden_states=True)

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    # train_doc = Document(json.load(open(args['train_file'])))
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, evaluation=False,
                             tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    vocab = train_batch.vocab
    # dev_doc = Document(json.load(open(args['eval_file'])))
    dev_batch = DataLoader(args['eval_file'], args['batch_size'], args, vocab=vocab,
                           evaluation=True, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    dev_gold_tags = dev_batch.tags

    print("TAG NER: ", vocab['tag']._unit2id)

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    print("Training tagger...")
    trainer = TrainerNER(args=args, vocab=vocab, config_phobert=config_phobert, use_cuda=args['cuda'])
    # print(trainer.model)

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
    # current_lr = trainer.optimizer.param_groups[0]['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    # LR scheduling
    # if args['lr_decay'] > 0:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='max', factor=args['lr_decay'], \
    #         patience=args['patience'], verbose=True, min_lr=args['min_lr'])
    # else:
    #     scheduler = None

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
    # start training
    train_loss = 0
    current_epoch = 0
    # while True:
    #     should_stop = False
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
            # if global_step % args['log_step'] == 0:
            #     duration = time.time() - start_time
            #     print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
            #             max_steps, loss, duration, current_lr))

            # if global_step % args['eval_interval'] == 0:
            #     # duration = time.time() - start_time
            #     # print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step, \
            #     #                         max_steps, loss, duration, current_lr))
            #     print("Evaluating on dev set...")
            #     dev_preds = []
            #     for batch in dev_batch:
            #         preds = trainer.predict(batch)
            #         dev_preds += preds
            #
            #     # print(dev_preds)
            #     # print(dev_gold_tags)
            #
            #     p, r, dev_score = scorer.score_by_entity(dev_preds, dev_gold_tags)
            #     print("step {}:, p_score = {:.4f}, r_score = {:.4f}, f1_score = {:.4f}".format(
            #         global_step, p, r, dev_score))
            #
            #     # save best model
            #     if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
            #         trainer.save(model_file)
            #         print("New best model saved.")
            #         best_dev_preds = dev_preds
            #         current_epoch = 0
            #
            #     dev_score_history += [dev_score]
            #     print("")

                # eval on dev
        print("Evaluating on dev set...")
        dev_preds = []
        for batch in dev_batch:
            preds = trainer.predict(batch)
            dev_preds += preds

        # print(dev_preds)
        # print(dev_gold_tags)

        p, r, dev_score = scorer.score_by_entity(dev_preds, dev_gold_tags)

        train_loss = train_loss / len(train_batch)  # avg loss per batch
        print("step {}: train_loss = {:.6f}, p_score = {:.4f}, r_score = {:.4f}, f1_score = {:.4f}".format(
            global_step, train_loss, p, r, dev_score))
        train_loss = 0

        # save best model
        if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
            trainer.save(model_file)
            print("New best model saved.")
            best_dev_preds = dev_preds
            current_epoch = 0

        dev_score_history += [dev_score]
        print("")

        # lr schedule
        # if scheduler is not None:
        #     scheduler.step(dev_score)
        #
        #     # check stopping
        # current_lr = trainer.optimizer.param_groups[0]['lr']
        # if current_lr < 1e-3:
        #     trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']),
        #                            eps=1e-6)
        # if global_step >= args['max_steps'] or current_lr <= args['min_lr']:
        #     should_stop = True
        #     break
        #
        # if should_stop:
        #     break

        train_batch.reshuffle()

    print("Training ended with {} steps.".format(global_step))

    best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
    print("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))


def evaluate(args):
    # file paths
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_nertagger.pt'.format(args['save_dir'], args['shorthand'])

    # pretrain = None
    # load model
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)
    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    config_phobert = AutoConfig.from_pretrained(args["phobert_model"], output_hidden_states=True)
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args["phobert_model"])
    print("Loading model from: {}".format(model_file))

    use_cuda = args['cuda'] and not args['cpu']
    trainer = TrainerNER(model_file=model_file, use_cuda=use_cuda, config_phobert=config_phobert)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    # for k in args:
    #     if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand', 'mode', 'scheme']:
    #         loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    # doc = Document(json.load(open(args['eval_file'])))
    batch = DataLoader(args['eval_file'], args['batch_size'], args, vocab=vocab, evaluation=True,
                       tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])

    print("Start evaluation...")
    preds = []
    for i, b in enumerate(batch):
        preds += trainer.predict(b)

    gold_tags = batch.tags
    # print(preds)
    # print(gold_tags)
    p, r, score = scorer.score_by_entity(preds, gold_tags)

    print("NER tagger score:")
    print("{} {:.2f} {:.2f} {:.2f}".format(args['shorthand'], p*100, r*100, score*100))


if __name__ == '__main__':
    main()
