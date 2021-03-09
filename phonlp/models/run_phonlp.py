# -*- coding: utf-8 -*-
"""
Training, evaluation and annotate for joint model.
"""

import argparse
import random
import sys

import numpy as np
import torch
from phonlp.annotate_model import JointModel
from phonlp.models.common import utils as util
from phonlp.models.common.doc import DEPREL, HEAD, Document
from phonlp.models.depparse import scorer as score_dep
from phonlp.models.jointmodel.data import BuildVocab, DataLoaderDep, DataLoaderNER, DataLoaderPOS
from phonlp.models.jointmodel.trainer import JointTrainer
from phonlp.models.ner import scorer as score_ner
from phonlp.models.ner.vocab import MultiVocab
from phonlp.models.pos import scorer as score_pos
from phonlp.utils.conll import CoNLL
from tqdm import tqdm
from transformers import AdamW, AutoConfig, AutoTokenizer, get_constant_schedule, get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file_dep",
        type=str,
        default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-train.conll",
        help="Input file for data loader.",
    )

    parser.add_argument(
        "--eval_file_dep",
        type=str,
        default="/home/ubuntu/linhnt140/data/VnDTv1.1_predictedPOS/VnDTv1.1-dev.conll",
        help="Input file for data loader.",
    )
    parser.add_argument("--output_file_dep", type=str, default="./jointmodel/dep.out", help="Output CoNLL-U file.")

    # POS
    parser.add_argument(
        "--train_file_pos",
        type=str,
        default="/home/ubuntu/linhnt140/data/POS_data/POS_data/VLSP2013_POS_train.txt",
        help="Input file for data loader.",
    )
    parser.add_argument(
        "--eval_file_pos",
        type=str,
        default="/home/ubuntu/linhnt140/data/POS_data/POS_data/VLSP2013_POS_dev.txt",
        help="Input file for data loader.",
    )

    # NER
    parser.add_argument(
        "--train_file_ner",
        type=str,
        default="/home/ubuntu/linhnt140/data/NER_data/train.txt",
        help="Input file for data loader.",
    )
    parser.add_argument(
        "--eval_file_ner",
        type=str,
        default="/home/ubuntu/linhnt140/data/NER_data/dev.txt",
        help="Input file for data loader.",
    )
    # Anotate corpus
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/ubuntu/linhnt140/data/annotate/input.txt",
        help="Input file for annotate corpus.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/ubuntu/linhnt140/data/annotate/output.txt",
        help="Output file for annotate corpus.",
    )

    parser.add_argument("--mode", default="train", choices=["train", "eval", "annotate"])
    parser.add_argument("--deep_biaff_hidden_dim", type=int, default=400)
    parser.add_argument("--tag_emb_dim", type=int, default=100)
    parser.add_argument("--word_dropout", type=float, default=0.33)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--linearization", type=bool, default=True, help="Turn off linearization term.")
    parser.add_argument("--distance", type=bool, default=True, help="Turn off linearization term.")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epoch", type=int, default=40)

    parser.add_argument("--lambda_pos", type=float, default=0.4, help="weight for pos loss.")
    parser.add_argument("--lambda_ner", type=float, default=0.2, help="weight for ner loss.")
    parser.add_argument("--lambda_dep", type=float, default=0.4, help="weight for dep loss.")

    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="saved_models/jointmodel", help="Root dir for saving models.")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
    # bert
    parser.add_argument("--pretrained_lm", type=str, default="vinai/phobert-base")
    parser.add_argument("--max_sequence_length", type=int, default=256)
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

    if args["mode"] == "train":
        train(args)
    elif args["mode"] == "eval":
        evaluate(args)
    else:
        annotate(
            input_file=args["input_file"], output_file=args["output_file"], args=args, batch_size=args["batch_size"]
        )


def train(args):
    util.ensure_dir(args["save_dir"])
    model_file = args["save_dir"] + "/" + "phonlp.pt"

    tokenizer = AutoTokenizer.from_pretrained(args["pretrained_lm"], use_fast=False)
    config_phobert = AutoConfig.from_pretrained(args["pretrained_lm"], output_hidden_states=True)

    print("Loading data with batch size {}...".format(args["batch_size"]))
    train_doc_dep = Document(CoNLL.conll2dict(input_file=args["train_file_dep"]))
    vocab = BuildVocab(args, args["train_file_pos"], train_doc_dep, args["train_file_ner"]).vocab

    train_batch_pos = DataLoaderPOS(
        args["train_file_pos"],
        args["batch_size"],
        args,
        vocab=vocab,
        evaluation=False,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )
    train_batch_dep = DataLoaderDep(
        train_doc_dep,
        args["batch_size"],
        args,
        vocab=vocab,
        evaluation=False,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )
    train_batch_ner = DataLoaderNER(
        args["train_file_ner"],
        args["batch_size"],
        args,
        vocab=vocab,
        evaluation=False,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )

    dev_doc_dep = Document(CoNLL.conll2dict(input_file=args["eval_file_dep"]))

    dev_batch_pos = DataLoaderPOS(
        args["eval_file_pos"],
        args["batch_size"],
        args,
        vocab=vocab,
        sort_during_eval=True,
        evaluation=True,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )
    dev_batch_dep = DataLoaderDep(
        dev_doc_dep,
        args["batch_size"],
        args,
        vocab=vocab,
        sort_during_eval=True,
        evaluation=True,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )
    dev_batch_ner = DataLoaderNER(
        args["eval_file_ner"],
        args["batch_size"],
        args,
        vocab=vocab,
        evaluation=True,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )

    # pred and gold path
    system_pred_file = args["output_file_dep"]
    gold_file = args["eval_file_dep"]

    # ##POS

    dev_gold_tags = dev_batch_ner.tags

    # skip training if the language does not have training or dev data
    if len(train_batch_pos) == 0 or len(dev_batch_pos) == 0:
        print("Skip training because no data available...")
        sys.exit(0)

    print("Training jointmodel...")
    trainer = JointTrainer(args, vocab, None, config_phobert, args["cuda"])
    # ###
    tsfm = trainer.model.phobert
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = True

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

    # Creating optimizer and lr schedulers
    param_optimizer = list(trainer.model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    num_train_optimization_steps = int(args["num_epoch"] * len(train_batch_pos) / args["accumulation_steps"])
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args["lr"], correct_bias=False
    )  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=num_train_optimization_steps
    )
    get_constant_schedule(optimizer)
    for epoch in range(args["num_epoch"]):
        ####
        optimizer.zero_grad()
        print(" EPOCH  : ", epoch)
        step = 0
        lambda_pos = args["lambda_pos"]
        lambda_ner = args["lambda_ner"]
        lambda_dep = args["lambda_dep"]

        epoch_size = max([len(train_batch_pos), len(train_batch_dep), len(train_batch_ner)])
        for i in tqdm(range(epoch_size)):
            step += 1
            global_step += 1
            batch_pos = train_batch_pos[i]
            batch_dep = train_batch_dep[i]
            batch_ner = train_batch_ner[i]
            ###
            loss, loss_pos, loss_ner, loss_dep = trainer.update(
                batch_dep, batch_pos, batch_ner, lambda_pos=lambda_pos, lambda_dep=lambda_dep, lambda_ner=lambda_ner
            )  # update step
            train_loss += loss
            train_loss_pos += loss_pos
            train_loss_dep += loss_dep
            train_loss_ner += loss_ner
            ###

            if i % args["accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if epoch_size == len(train_batch_pos):
                if step % len(train_batch_dep) == 0:
                    train_batch_dep.reshuffle()
                if step % len(train_batch_ner) == 0:
                    train_batch_ner.reshuffle()
            elif epoch_size == len(train_batch_ner):
                if step % len(train_batch_dep) == 0:
                    train_batch_dep.reshuffle()
                if step % len(train_batch_pos) == 0:
                    train_batch_pos.reshuffle()
            elif epoch_size == len(train_batch_dep):
                if step % len(train_batch_pos) == 0:
                    train_batch_dep.reshuffle()
                if step % len(train_batch_ner) == 0:
                    train_batch_ner.reshuffle()
            if step % args["eval_interval"] == 0:
                print("Evaluating on dev set...")
                dev_preds_dep = []
                dev_preds_upos = []
                dev_preds_ner = []
                for batch in dev_batch_dep:
                    preds_dep = trainer.predict_dep(batch)
                    dev_preds_dep += preds_dep
                ###
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

                print(
                    "step {}: dev_las_score = {:.4f}, dev_uas_score = {:.4f}, dev_pos = {:.4f}, dev_ner_p = {:.4f}, dev_ner_r = {:.4f}, dev_ner_f1 = {:.4f}".format(
                        global_step, las_dev, uas_dev, accuracy_pos_dev, p, r, f1
                    )
                )

                # save best model
                if las_dev + accuracy_pos_dev + f1 >= (las_score_history + upos_score_history + f1_score_history):
                    las_score_history = las_dev
                    upos_score_history = accuracy_pos_dev
                    uas_score_history = uas_dev
                    f1_score_history = f1
                    trainer.save(model_file)
                    print("new best model saved.")
                print("")

        print("Evaluating on dev set...")
        dev_preds_dep = []
        dev_preds_upos = []
        dev_preds_ner = []
        for batch in dev_batch_dep:
            preds_dep = trainer.predict_dep(batch)
            dev_preds_dep += preds_dep

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

        train_loss = train_loss / len(train_batch_pos)  # avg loss per batch
        train_loss_dep = train_loss_dep / len(train_batch_pos)
        train_loss_pos = train_loss_pos / len(train_batch_pos)
        train_loss_ner = train_loss_ner / len(train_batch_pos)

        print(
            "step {}: train_loss = {:.6f}, train_loss_dep = {:.6f}, train_loss_pos = {:.6f}, train_loss_ner = {:.6f}, dev_las_score = {:.4f}, dev_uas_score = {:.4f}, dev_pos = {:.4f}, dev_ner_p = {:.4f}, dev_ner_r = {:.4f}, dev_ner_f1 = {:.4f} ".format(
                global_step,
                train_loss,
                train_loss_dep,
                train_loss_pos,
                train_loss_ner,
                las_dev,
                uas_dev,
                accuracy_pos_dev,
                p,
                r,
                f1,
            )
        )

        # save best model
        if las_dev + accuracy_pos_dev + f1 >= (las_score_history + upos_score_history + f1_score_history):
            las_score_history = las_dev
            upos_score_history = accuracy_pos_dev
            uas_score_history = uas_dev
            f1_score_history = f1
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

    print("Training ended with {} epochs.".format(epoch))

    best_las, uas, upos, f1 = (
        las_score_history * 100,
        uas_score_history * 100,
        upos_score_history * 100,
        f1_score_history * 100,
    )
    print("Best dev las = {:.2f}, uas = {:.2f}, upos = {:.2f}, f1 = {:.2f}".format(best_las, uas, upos, f1))


def evaluate(args):
    # file paths
    system_pred_file = args["output_file_dep"]
    gold_file = args["eval_file_dep"]
    model_file = args["save_dir"] + "/" + "phonlp.pt"

    checkpoint = torch.load(model_file, lambda storage, loc: storage)
    loaded_args = checkpoint["config"]
    vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
    config_phobert = AutoConfig.from_pretrained(loaded_args["pretrained_lm"], output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(loaded_args["pretrained_lm"], use_fast=False)

    # load model
    print("Loading model from: {}".format(model_file))
    use_cuda = args["cuda"] and not args["cpu"]
    trainer = JointTrainer(model_file=model_file, use_cuda=use_cuda, config_phobert=config_phobert)

    # load data
    print("Loading data with batch size {}...".format(args["batch_size"]))

    test_doc_dep = Document(CoNLL.conll2dict(input_file=args["eval_file_dep"]))

    test_batch_pos = DataLoaderPOS(
        args["eval_file_pos"],
        args["batch_size"],
        args,
        vocab=vocab,
        sort_during_eval=True,
        evaluation=True,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )
    test_batch_dep = DataLoaderDep(
        test_doc_dep,
        args["batch_size"],
        args,
        vocab=vocab,
        sort_during_eval=True,
        evaluation=True,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )
    test_batch_ner = DataLoaderNER(
        args["eval_file_ner"],
        args["batch_size"],
        args,
        vocab=vocab,
        evaluation=True,
        tokenizer=tokenizer,
        max_seq_length=args["max_sequence_length"],
    )

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

    print(
        "{} POS tagging: {:.2f}, NER: {:.2f}, Dependency parsing: {:.2f}/{:.2f}".format(
            "Evaluation results: ", accuracy_pos * 100, f1 * 100, las * 100, uas * 100
        )
    )


def annotate(input_file=None, output_file=None, args=None, batch_size=1):
    model_file = args["save_dir"] + "/" + "phonlp.pt"
    print("Loading model from: {}".format(model_file))
    checkpoint = torch.load(model_file, lambda storage, loc: storage)
    args = checkpoint["config"]
    vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args["pretrained_lm"], use_fast=False)
    config_phobert = AutoConfig.from_pretrained(args["pretrained_lm"], output_hidden_states=True)
    model = JointModel(args, vocab, config_phobert, tokenizer)
    model.load_state_dict(checkpoint["model"], strict=False)
    if torch.cuda.is_available() is False:
        model.to(torch.device("cpu"))
    else:
        model.to(torch.device("cuda"))
    model.eval()
    model.annotate(input_file=input_file, output_file=output_file, batch_size=batch_size)


if __name__ == "__main__":
    main()
