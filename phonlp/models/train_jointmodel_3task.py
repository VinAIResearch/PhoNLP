# -*- coding: utf-8 -*-
"""
Training and evaluation for joint model.
"""


from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
from transformers import *
from models.common.doc import *
from utils.conll import CoNLL
from transformers.modeling_utils import *
from models.common import utils as util
from models.ner import scorer as score_ner
from models.pos import scorer as score_pos
from models.depparse import scorer as score_dep
from models.jointmodel3task.trainer import TrainerJoint
from models.jointmodel3task.data import DataLoaderDep, DataLoaderPOS, BuildVocab, DataLoaderNER
import os
import shutil
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch
from torch import nn, optim
import sys
sys.path.append('../')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_dep', type=str,
                        default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-train.conll", help='Input file for data loader.')

    parser.add_argument('--eval_file_dep', type=str,
                        default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-dev.conll", help='Input file for data loader.')
    parser.add_argument('--output_file_dep', type=str, default="./jointmodel3task/dep.out", help='Output CoNLL-U file.')

    # POS
    parser.add_argument('--train_file_pos', type=str,
                        default="/home/ubuntu/linhnt140/data/POS_data/POS_data/VLSP2013_POS_train.txt",
                        help='Input file for data loader.')
    parser.add_argument('--eval_file_pos', type=str,
                        default="/home/ubuntu/linhnt140/data/POS_data/POS_data/VLSP2013_POS_dev.txt",
                        help='Input file for data loader.')

    # NER
    parser.add_argument('--train_file_ner', type=str,
                        default="/home/ubuntu/linhnt140/data/NER_data/train.txt",
                        help='Input file for data loader.')
    parser.add_argument('--eval_file_ner', type=str,
                        default="/home/ubuntu/linhnt140/data/NER_data/dev.txt",
                        help='Input file for data loader.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, default="vi", help='Language')
    parser.add_argument('--shorthand', type=str, default="VnDTv1.1", help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=100)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=100)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=100)
    parser.add_argument('--transformed_dim', type=int, default=300)

    parser.add_argument('--scheme', type=str, default='bioes', help="The tagging scheme to use: bio or bioes.")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    #parser.add_argument('--no_pretrain', dest='pretrain', action='store_true', help="Turn off pretrained embeddings.")
    # parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--linearization', type=bool, default=True, help="Turn off linearization term.")
    parser.add_argument('--pretrain', type=bool, default=False, help="Turn off pretrained embeddings.")
    parser.add_argument('--distance', type=bool, default=True, help="Turn off linearization term.")

    #parser.add_argument('--no_distance', dest='distance', action='store_true', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    #parser.add_argument('--optim', type=str, default='adamax', help='sgd, adagrad, adamw, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate to stop training.')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--num_epoch', type=int, default=40)
    parser.add_argument('--lr_decay', type=float, default=0.5, help="LR decay rate.")
    parser.add_argument('--patience', type=int, default=2, help="Patience for LR decay.")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')

    parser.add_argument('--lambda_pos', type=float, default=0.4, help="weight for pos loss.")
    parser.add_argument('--lambda_ner', type=float, default=0.2, help="weight for ner loss.")
    parser.add_argument('--lambda_dep', type=float, default=0.4, help="weight for dep loss.")

    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--max_steps_before_stop', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models_newversion/jointmodel',
                        help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")
    parser.add_argument('--logfile_name', type=str, default='logfile_jointmodel3task.txt',
                        help="File name to save the model")

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
        else '{}/{}_jointmodel.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors if needed
    # pretrain = None
    # if args['pretrain']:
    #     # vec_file = util.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
    #     vec_file = args['wordvec_dir'] + "word2vec_vi_words_300dims.txt"
    #     pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    #     pretrain = Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])

    # load data
    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    # #config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args['phobert_model'], use_fast=False)
    config_phobert = AutoConfig.from_pretrained(args['phobert_model'], output_hidden_states=True)

    print("Loading data with batch size {}...".format(args['batch_size']))
    # Document( list of list of dict ( dict is a token))
    train_doc_dep = Document(CoNLL.conll2dict(input_file=args['train_file_dep']))
    # train_doc_pos = Document(CoNLL.conll2dict(input_file=args['train_file_pos']))
    vocab = BuildVocab(args, args['train_file_pos'], train_doc_dep, args['train_file_ner']).vocab

    train_batch_pos = DataLoaderPOS(args['train_file_pos'], args['batch_size'], args, vocab=vocab,
                                    evaluation=False, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    train_batch_dep = DataLoaderDep(train_doc_dep, args['batch_size'], args, vocab=vocab,
                                    evaluation=False, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])
    train_batch_ner = DataLoaderNER(args['train_file_ner'], args['batch_size'], args, vocab=vocab,
                                    evaluation=False, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])

    print("VOCAB SIZE POS IN POS DATASET: ", list(vocab['upos']))
    print("VOCAB SIZE NER IN NER DATASET: ", list(vocab['ner_tag']))
    # Document( list of list of dict ( dict is a token))
    dev_doc_dep = Document(CoNLL.conll2dict(input_file=args['eval_file_dep']))
    # dev_doc_pos = Document(CoNLL.conll2dict(input_file=args['eval_file_pos']))

    # test_doc_dep = Document(CoNLL.conll2dict(input_file=args['eval_file_test']))  ### Document( list of list of dict ( dict is a token))
    # test_doc_pos = Document(CoNLL.conll2dict(input_file=args['eval_file_pos_test']))

    dev_batch_pos = DataLoaderPOS(args['eval_file_pos'], args['batch_size'], args, vocab=vocab, sort_during_eval=True,
                                  evaluation=True, tokenizer=tokenizer,
                                  max_seq_length=args['max_sequence_length'])
    dev_batch_dep = DataLoaderDep(dev_doc_dep, args['batch_size'], args, vocab=vocab,
                                  sort_during_eval=True,
                                  evaluation=True, tokenizer=tokenizer,
                                  max_seq_length=args['max_sequence_length'])
    dev_batch_ner = DataLoaderNER(args['eval_file_ner'], args['batch_size'], args, vocab=vocab,
                                  evaluation=True, tokenizer=tokenizer, max_seq_length=args['max_sequence_length'])

    # pred and gold path
    system_pred_file = args['output_file_dep']
    gold_file = args['eval_file_dep']

    # ##POS

    dev_gold_tags = dev_batch_ner.tags

    # skip training if the language does not have training or dev data
    if len(train_batch_pos) == 0 or len(dev_batch_pos) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    print("Training jointmodel...")
    trainer = TrainerJoint(args, vocab, None, config_phobert, args['cuda'])
    # ###
    tsfm = trainer.model.phobert
    tq = tqdm(range(args['num_epoch'] + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = True
    frozen = True
    # ue
    # trainer.model.cuda()
    # if torch.cuda.device_count() > 1:
    #     trainer.model = nn.DataParallel(trainer.model, device_ids=[0,1])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # trainer.model.to(device)

    global_step = 0
    las_score_history = 0
    uas_score_history = 0
    upos_score_history = 0
    f1_score_history = 0
    ####

    # start training
    train_loss = 0
    train_loss_pos = 0
    train_loss_dep = 0
    train_loss_ner = 0
    # log_file = open(args['save_dir'] + '/' + args['logfile_name'], 'w')
    # ###
    # parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    # optimizer = util.get_optimizer(args['optim'], parameters, args['lr'], betas=(0.9, args['beta2']), eps=1e-6)
    #
    # if args['lr_decay'] > 0:
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args['lr_decay'], \
    #         patience=args['patience'], verbose=True, min_lr=args['min_lr'])
    # else:
    #     scheduler = None

    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
    # self.optimizer.step()

    # Creating optimizer and lr schedulers
    param_optimizer = list(trainer.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = int(args['num_epoch'] * len(train_batch_pos) * args['accumulation_steps'])
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['lr'],
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5,
                                                num_training_steps=num_train_optimization_steps)
    scheduler0 = get_constant_schedule(optimizer)
    # current_lr = args['lr']
    # count_step = 0
    for epoch in range(args['num_epoch']):
        # if epoch > 0 and frozen:
        #     for child in tsfm.children():
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     frozen = False
        #     del scheduler0
        #     torch.cuda.empty_cache()

        ####
        optimizer.zero_grad()
        print(" EPOCH  : ", epoch)
        # log_file.write("EPOCH: " + str(epoch) + '\n')
        step = 0
        lambda_pos = args['lambda_pos']
        lambda_ner = args['lambda_ner']
        lambda_dep = args['lambda_dep']

        pbar = tqdm(enumerate(train_batch_pos), total=len(train_batch_pos), leave=False)
        for i, batch_pos in pbar:
            step += 1
            global_step += 1
            batch_dep = train_batch_dep[i]
            batch_ner = train_batch_ner[i]
            ###
            loss, loss_pos, loss_ner, loss_dep = trainer.update(
                batch_dep, batch_pos, batch_ner, eval=False, lambda_pos=lambda_pos, lambda_dep=lambda_dep, lambda_ner=lambda_ner)  # update step
            train_loss += loss
            train_loss_pos += loss_pos
            train_loss_dep += loss_dep
            train_loss_ner += loss_ner
            ###

            if i % args['accumulation_steps'] == 0:
                # torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), args['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                # if not frozen:
                scheduler.step()
                # else:
                #     scheduler0.step()

            if step % len(train_batch_dep) == 0:
                train_batch_dep.reshuffle()
            if step % len(train_batch_ner) == 0:
                train_batch_ner.reshuffle()
            # if step % len(train_batch_dep) == 0 or step % len(train_batch_ner) == 0: #
            if step % args['eval_interval'] == 0:
                print("Evaluating on dev set...")
                # count_step += 1
                dev_preds_dep = []
                dev_preds_upos = []
                dev_preds_ner = []
                for batch in dev_batch_dep:
                    preds_dep = trainer.predict_dep(batch)
                    dev_preds_dep += preds_dep
                ###
                #dev_preds_dep = dev_preds_dep[:200]
                dev_preds_dep = util.unsort(dev_preds_dep, dev_batch_dep.data_orig_idx_dep)
                dev_batch_dep.doc_dep.set([HEAD, DEPREL], [y for x in dev_preds_dep for y in x])
                CoNLL.dict2conll(dev_batch_dep.doc_dep.to_dict(), system_pred_file)
                _, _, las_dev, uas_dev = score_dep.score(system_pred_file, gold_file)

                for batch in dev_batch_pos:
                    preds_pos = trainer.predict_pos(batch)
                    dev_preds_upos += preds_pos
                dev_preds_upos = util.unsort(dev_preds_upos, dev_batch_pos.data_orig_idx_pos)
                # dev_batch.doc.set([UPOS], [y for x in dev_preds_upos for y in x])
                # CoNLL.dict2conll(dev_batch.doc.to_dict(), system_pred_file_pos)
                # _, _, accuracy_pos_dev = score_pos.score(system_pred_file_pos, gold_file_pos)
                accuracy_pos_dev = score_pos.score_acc(dev_preds_upos, dev_batch_pos.upos)

                for batch in dev_batch_ner:
                    preds_ner = trainer.predict_ner(batch)
                    dev_preds_ner += preds_ner
                p, r, f1 = score_ner.score_by_entity(dev_preds_ner, dev_gold_tags)
                for i in range(len(dev_batch_ner)):
                    assert len(dev_preds_ner[i]) == len(dev_gold_tags[i])
                # print("DEV_PREDS_NER: ", dev_preds_ner)
                # print("DEV_GOLD_NER: ", dev_gold_tags)

                print(
                    "step {}: dev_las_score = {:.4f}, dev_uas_score = {:.4f}, dev_pos = {:.4f}, dev_ner_p = {:.4f}, dev_ner_r = {:.4f}, dev_ner_f1 = {:.4f}".format(
                        global_step, las_dev, uas_dev, accuracy_pos_dev, p, r, f1))

                # if scheduler is not None:
                #     scheduler.step(las_dev + accuracy_pos_dev + f1)

                # save best model
                if las_dev + accuracy_pos_dev + f1 >= (las_score_history + upos_score_history + f1_score_history):
                    las_score_history = las_dev
                    upos_score_history = accuracy_pos_dev
                    uas_score_history = uas_dev
                    f1_score_history = f1
                    last_best_step = global_step
                    trainer.save(model_file)
                    print("new best model saved.")
                    # count_step = 0
                    # best_dev_preds = dev_preds
                #
                print("")

        print("Evaluating on dev set...")
        # count_step += 1
        dev_preds_dep = []
        dev_preds_upos = []
        dev_preds_ner = []
        for batch in dev_batch_dep:
            preds_dep = trainer.predict_dep(batch)
            dev_preds_dep += preds_dep
        ###
        # dev_preds_dep = dev_preds_dep[:200]
        dev_preds_dep = util.unsort(dev_preds_dep, dev_batch_dep.data_orig_idx_dep)
        dev_batch_dep.doc_dep.set([HEAD, DEPREL], [y for x in dev_preds_dep for y in x])
        CoNLL.dict2conll(dev_batch_dep.doc_dep.to_dict(), system_pred_file)
        _, _, las_dev, uas_dev = score_dep.score(system_pred_file, gold_file)

        for batch in dev_batch_pos:
            preds_pos = trainer.predict_pos(batch)
            dev_preds_upos += preds_pos
        dev_preds_upos = util.unsort(dev_preds_upos, dev_batch_pos.data_orig_idx_pos)
        accuracy_pos_dev = score_pos.score_acc(dev_preds_upos, dev_batch_pos.upos)

        for batch in dev_batch_ner:
            preds_ner = trainer.predict_ner(batch)
            dev_preds_ner += preds_ner
        p, r, f1 = score_ner.score_by_entity(dev_preds_ner, dev_gold_tags)
        for i in range(len(dev_batch_ner)):
            assert len(dev_preds_ner[i]) == len(dev_gold_tags[i])
        # print("DEV_PRED_NER: ", dev_preds_ner)
        # print("GOLD_TAG: ", dev_gold_tags)

        train_loss = train_loss / len(train_batch_pos)  # avg loss per batch
        train_loss_dep = train_loss_dep / len(train_batch_pos)
        train_loss_pos = train_loss_pos / len(train_batch_pos)
        train_loss_ner = train_loss_ner / len(train_batch_pos)

        print(
            "step {}: train_loss = {:.6f}, train_loss_dep = {:.6f}, train_loss_pos = {:.6f}, train_loss_ner = {:.6f}, dev_las_score = {:.4f}, dev_uas_score = {:.4f}, dev_pos = {:.4f}, dev_ner_p = {:.4f}, dev_ner_r = {:.4f}, dev_ner_f1 = {:.4f} ".format(
                global_step, train_loss, train_loss_dep, train_loss_pos, train_loss_ner, las_dev, uas_dev, accuracy_pos_dev, p, r, f1))

        # save best model
        if las_dev + accuracy_pos_dev + f1 >= (las_score_history + upos_score_history + f1_score_history):
            las_score_history = las_dev
            upos_score_history = accuracy_pos_dev
            uas_score_history = uas_dev
            f1_score_history = f1
            last_best_step = global_step
            trainer.save(model_file)
            print("new best model saved.")
        train_loss = 0
        train_loss_pos = 0
        train_loss_dep = 0
        train_loss_ner = 0

        print("")
        train_batch_dep.reshuffle()
        train_batch_pos.reshuffle()
        train_batch_ner.reshuffle()
        # train_batch.reshuffle('ner')
        # if count_step >= 5 and current_lr > args['min_lr']:
        #     current_lr = args['lr'] / 2
        #     count_step = 0
        #     optimizer = AdamW(optimizer_grouped_parameters, lr=current_lr, correct_bias=False)

    print("Training ended with {} epochs.".format(epoch))
    # log_file.write("Training ended with {} epochs.".format(epoch))
    # log_file.write('\n')

    best_las, uas, upos, f1 = las_score_history*100, uas_score_history * 100, upos_score_history * 100, f1_score_history * 100
    print("Best dev las = {:.2f}, uas = {:.2f}, upos = {:.2f}, f1 = {:.2f}".format(best_las, uas, upos, f1))
    # log_file.write("Best dev las = {:.2f}, uas = {:.2f}, upos = {:.2f}, f1 = {:.2f}".format(best_las, uas, upos, f1))
    # log_file.close()


def evaluate(args):
    # file paths
    system_pred_file = args['output_file_dep']
    gold_file = args['eval_file_dep']
    # system_pred_file_pos = args['output_file_pos_test']
    # gold_file_pos = args['gold_file_pos_test']
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_jointmodel.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrain; note that we allow the pretrain_file to be non-existent
    # pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
    # pretrain = Pretrain(pretrain_file)
    # config_phobert = RobertaConfig.from_pretrained(args['config_path'], output_hidden_states=True)
    # vocab_phobert = Dictionary()
    # vocab_phobert.add_from_file(args['dict_path'])
    # args_lib = parse_args()
    # bpe = fastBPE(args_lib)
    # pretrain = None
    config_phobert = AutoConfig.from_pretrained(args["phobert_model"], output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args["phobert_model"])

    # load model
    print("Loading model from: {}".format(model_file))
    use_cuda = args['cuda'] and not args['cpu']
    trainer = TrainerJoint(model_file=model_file, use_cuda=use_cuda, config_phobert=config_phobert)
    loaded_args, vocab = trainer.args, trainer.vocab
    print(loaded_args)
    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
            loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))

    test_doc_dep = Document(
        CoNLL.conll2dict(input_file=args['eval_file_dep']))  # Document( list of list of dict ( dict is a token))
    # test_doc_pos = Document(CoNLL.conll2dict(input_file=args['eval_file_pos']))

    test_batch_pos = DataLoaderPOS(args['eval_file_pos'], args['batch_size'], args, vocab=vocab,
                                   sort_during_eval=True,
                                   evaluation=True, tokenizer=tokenizer,
                                   max_seq_length=args['max_sequence_length'])
    test_batch_dep = DataLoaderDep(test_doc_dep, args['batch_size'], args, vocab=vocab,
                                   sort_during_eval=True,
                                   evaluation=True, tokenizer=tokenizer,
                                   max_seq_length=args['max_sequence_length'])
    test_batch_ner = DataLoaderNER(args['eval_file_ner'], args['batch_size'], args, vocab=vocab,
                                   evaluation=True, tokenizer=tokenizer,
                                   max_seq_length=args['max_sequence_length'])

    print("Start evaluation...")
    test_preds_dep = []
    test_preds_upos = []
    test_preds_ner = []
    for batch in test_batch_dep:
        preds_dep = trainer.predict_dep(batch)
        test_preds_dep += preds_dep
    test_preds_dep = util.unsort(test_preds_dep, test_batch_dep.data_orig_idx_dep)
    test_batch_dep.doc_dep.set([HEAD, DEPREL], [y for x in test_preds_dep for y in x])
    CoNLL.dict2conll(test_batch_dep.doc_dep.to_dict(), system_pred_file)
    _, _, las, uas = score_dep.score(system_pred_file, gold_file)

    for batch in test_batch_pos:
        preds_pos = trainer.predict_pos(batch)
        test_preds_upos += preds_pos
    test_preds_upos = util.unsort(test_preds_upos, test_batch_pos.data_orig_idx_pos)
    accuracy_pos = score_pos.score_acc(test_preds_upos, test_batch_pos.upos)

    for batch in test_batch_ner:
        preds_ner = trainer.predict_ner(batch)
        test_preds_ner += preds_ner
    p, r, f1 = score_ner.score_by_entity(test_preds_ner, test_batch_ner.tags)

    print("Test score:")
    print("{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
        args['shorthand'], las*100, uas * 100, accuracy_pos * 100, p * 100, r * 100, f1 * 100))


if __name__ == '__main__':
    main()
