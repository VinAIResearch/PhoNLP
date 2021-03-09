"""
Utils and wrappers for scoring taggers.
"""
import logging

from phonlp.models.common.utils import ud_scores


logger = logging.getLogger("PhoToolkit")


def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for tagger scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation["AllTags"]
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ["UPOS", "XPOS", "UFeats", "AllTags"]]
        logger.info("UPOS\tXPOS\tUFeats\tAllTags")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f


def score_acc(pred_pos, gold_pos):
    """ Wrapper for tagger scorer. """
    assert len(pred_pos) == len(gold_pos)
    num_words = 0
    correct = 0
    for i in range(len(pred_pos)):
        assert len(pred_pos[i]) == len(gold_pos[i])
        for j in range(len(pred_pos[i])):
            num_words += 1
            if str(pred_pos[i][j][0]) == str(gold_pos[i][j]):
                correct += 1
    logger.info("POS")
    logger.info("{:.2f}".format(correct / num_words))
    return correct / num_words
