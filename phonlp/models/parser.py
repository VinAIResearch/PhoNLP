# -*- coding: utf-8 -*-
"""
Entry point for training and evaluating a dependency parser.

This implementation combines a deep biaffine graph-based parser with linearization and distance features.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

"""
Training and evaluation for the parser.
"""
from torch import nn, optim
import torch
import random
import numpy as np
import argparse
from datetime import datetime
import time
import shutil
import os
from utils.conll import CoNLL
from models.depparse.data import DataLoaderDep
from models.depparse.trainer import TrainerDep
from models.depparse import scorer
from models.common import utils as util
from models.common.doc import *
from transformers.modeling_utils import *
from transformers import *
from tqdm import tqdm
import sys
sys.path.append('../')
#from PhoToolkit.models.common.pretrain import Pretrain
#from PhoToolkit.models import _training_logging
# from fairseq.data.encoders.fastbpe import fastBPE
# from fairseq.data import Dictionary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-predPOS-singlemodel-train.conll",
                        help='Input file for data loader.')

    parser.add_argument('--eval_file', type=str,
                        default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-predPOS-singlemodel-dev.conll",
                        help='Input file for data loader.')
    parser.add_argument('--output_file', type=str,
                        default="./depparse/dep.out",
                        help='Output CoNLL-U file.')
    # parser.add_argument('--gold_file', type=str,
    #                     default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-predPOS-dev.conll",
    #                     help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, default="vi", help='Language')
    parser.add_argument('--shorthand', type=str, default="VnDTv1.1", help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=100)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    # parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--linearization', type=bool, default=True, help="Turn off linearization term.")

    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adamax', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--num_epoch', type=int, default=40)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models_newversion/depparse',
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
    parser.add_argument('--bpe-codes', default="/home/ubuntu/linhnt140/PhoBERT_base_transformers/bpe.codes",
                        type=str, help='path to fastBPE BPE')
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
    print("Running parser in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)


def train(args):
    util.ensure_dir(args['save_dir'])
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors if needed
    # pretrain = None
    # if args['pretrain']:
    #     # vec_file = utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    #     vec_file = args['wordvec_dir'] + "word2vec_vi_words_300dims.txt"
    #     pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    #     pretrain = Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])

    # load data
    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)

    tokenizer = AutoTokenizer.from_pretrained(args['phobert_model'], use_fast=False)
    config_phobert = AutoConfig.from_pretrained(args['phobert_model'], output_hidden_states=True)

    print("Loading data with batch size {}...".format(args['batch_size']))
    # Document( list of list of dict ( dict is a token))
    train_doc = Document(CoNLL.conll2dict(input_file=args['train_file']))
    train_batch = DataLoaderDep(train_doc, args['batch_size'], args, evaluation=False,
                                tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    print("Number of train example: ", len(train_batch))
    vocab = train_batch.vocab
    dev_doc = Document(CoNLL.conll2dict(input_file=args['eval_file']))
    dev_batch = DataLoaderDep(dev_doc, args['batch_size'], args, vocab=vocab, evaluation=True,
                              sort_during_eval=True, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['eval_file']

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    print("Training parser...")
    trainer = TrainerDep(args, vocab, None, config_phobert, args['cuda'])
    ###
    tsfm = trainer.model.phobert
    tq = tqdm(range(args['num_epoch'] + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = True
    frozen = True
    ####
    global_step = 0
    max_steps = args['max_steps']
    las_score_history = []
    uas_score_history = []
    upos_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    #optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)
    ########
    parameters = list(trainer.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(args['num_epoch'] * len(train_batch) / args['accumulation_steps'])
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['lr'],
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    scheduler0 = get_constant_schedule(optimizer)
    # while True:
    #     do_break = False
    for epoch in range(args['num_epoch']):
        ####
        # if epoch > 0 and frozen:
        #     for child in tsfm.children():
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     frozen = False
        #     del scheduler0
        #     torch.cuda.empty_cache()
        # #####
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_batch), total=len(train_batch), leave=False)
        for i, batch in pbar:
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False)  # update step
            train_loss += loss
            # if global_step % args['log_step'] == 0:
            #     duration = time.time() - start_time
            #     print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
            #             max_steps, loss, duration, current_lr))

            # if global_step % args['eval_interval'] == 0:
            #     # eval on dev
            if i % args['accumulation_steps'] == 0 or i == (len(pbar) - 1):
                # torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), args['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                # if not frozen:
                scheduler.step()
                # else:
                #     scheduler0.step()
        print("Evaluating on dev set...")
        dev_preds = []
        # dev_preds_upos = []
        for batch in dev_batch:
            #print("BATCHHHHHHHHH: ", batch)
            preds = trainer.predict(batch)
            # dev_preds_upos += preds_upos
            dev_preds += preds
        dev_preds = util.unsort(dev_preds, dev_batch.data_orig_idx)
        # dev_preds_upos = utils.unsort(dev_preds_upos, dev_batch.data_orig_idx)

        dev_batch.doc_dep.set([HEAD, DEPREL], [y for x in dev_preds for y in x])
        # dev_batch.doc.set([UPOS], [y for x in dev_preds_upos for y in x])
        CoNLL.dict2conll(dev_batch.doc_dep.to_dict(), system_pred_file)
        # CoNLL.dict2conll(dev_batch.doc.to_dict(), system_pred_file)
        _, _, las, uas = scorer.score(system_pred_file, gold_file)

        # train_loss = train_loss / args['eval_interval'] # avg loss per batch
        train_loss = train_loss / len(train_batch)
        print("step {}: train_loss = {:.6f}, dev_las_score = {:.4f}, dev_uas_score = {:.4f}".format(
            global_step, train_loss, las, uas))
        train_loss = 0

        # save best model
        if len(las_score_history) == 0 or las > max(las_score_history):
            last_best_step = global_step
            trainer.save(model_file)
            print("new best model saved.")
            best_dev_preds = dev_preds

        las_score_history += [las]
        uas_score_history += [uas]
        # upos_score_history += [upos]
        print("")
        #
        # if global_step - last_best_step >= args['max_steps_before_stop']:
        #     if not using_amsgrad:
        #         print("Switching to AMSGrad")
        #         last_best_step = global_step
        #         using_amsgrad = True
        #         trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-6)
        #     # else:
        #     #do_break = True
        #     break

        #     if global_step >= args['max_steps']:
        #         do_break = True
        #         break
        #
        # if do_break: break

        train_batch.reshuffle()

    print("Training ended with {} epochs.".format(epoch))

    best_las, uas, best_eval = max(las_score_history)*100, max(uas_score_history) * 100, np.argmax(las_score_history)+1
    print("Best dev las = {:.2f}, uas = {:.2f}, at epoch = {}".format(best_las, uas, best_eval))


def evaluate(args):
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['eval_file']
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrain; note that we allow the pretrain_file to be non-existent
    # pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    # pretrain = Pretrain(pretrain_file)
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)
    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    pretrain = None
    config_phobert = AutoConfig.from_pretrained(args["phobert_model"], output_hidden_states=True)
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args["phobert_model"])
    # load model
    print("Loading model from: {}".format(model_file))
    use_cuda = args['cuda'] and not args['cpu']
    trainer = TrainerDep(model_file=model_file, use_cuda=use_cuda, config_phobert=config_phobert)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    # for k in args:
    #     if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
    #         loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    doc = Document(CoNLL.conll2dict(input_file=args['eval_file']))
    dev_batch = DataLoaderDep(doc, args['batch_size'], loaded_args, vocab=vocab, evaluation=True,
                              sort_during_eval=True, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])

    if len(dev_batch) > 0:
        print("Start evaluation...")
        # preds = []
        # for i, b in enumerate(batch):
        #     preds += trainer.predict(b)
        dev_preds = []
        # dev_preds_upos = []
        for i, batch in enumerate(dev_batch):
            # print("BATCHHHHHHHHH: ", batch)
            preds = trainer.predict(batch)
            # dev_preds_upos += preds_upos
            dev_preds += preds
    else:
        # skip eval if dev data does not exist
        preds = []
    # preds = utils.unsort(preds, batch.data_orig_idx)

    dev_preds = util.unsort(dev_preds, dev_batch.data_orig_idx)
    # dev_preds_upos = utils.unsort(dev_preds_upos, dev_batch.data_orig_idx)

    dev_batch.doc_dep.set([HEAD, DEPREL], [y for x in dev_preds for y in x])
    # dev_batch.doc.set([UPOS], [y for x in dev_preds_upos for y in x])
    CoNLL.dict2conll(dev_batch.doc_dep.to_dict(), system_pred_file)

    # CoNLL.dict2conll(dev_batch.doc.to_dict(), system_pred_file)
    _, _, las, uas = scorer.score(system_pred_file, gold_file)
    # print(dev_preds)
    # print(dev_batch.head)
    # print(dev_batch.deprel)

    # write to file and score
    # batch.doc.set([HEAD, DEPREL], [y for x in preds for y in x])
    # CoNLL.dict2conll(batch.doc.to_dict(), system_pred_file)

    # if gold_file is not None:
    #     _, _, score, uas = scorer.score(system_pred_file, gold_file)

    print("Parser score:")
    print("{} {:.2f} {:.2f}".format(args['shorthand'], las*100, uas * 100))


if __name__ == '__main__':
    main()
