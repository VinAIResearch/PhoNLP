"""
Utils and wrappers for scoring parsers.
"""
import logging

from phonlp.models.common.utils import ud_scores


logger = logging.getLogger("PhoNLPToolkit")


def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation["LAS"]
    el2 = evaluation["UAS"]
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ["UAS", "LAS"]]
        logger.info("UAS\tLAS")
        logger.info("{:.2f}\t{:.2f}".format(*scores))
    return p, r, f, el2.f1
