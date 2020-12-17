# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import sys
from PhoToolkit.models.common.vocab import VOCAB_PREFIX
from PhoToolkit.models.common.constant import lang2lcode
from PhoToolkit.models.pos.vocab import XPOSVocab, WordVocab
from PhoToolkit.models.common.doc import *
from PhoToolkit.utils.conll import CoNLL

if len(sys.argv) != 3:
    print('Usage: {} list_of_tb_file output_factory_file'.format(sys.argv[0]))
    sys.exit(0)

# Read list of all treebanks of concern
list_of_tb_file, output_file = sys.argv[1:]


def treebank_to_short_name(treebank):
    """ Convert treebank name to short code. """
    if treebank.startswith('UD_'):
        treebank = treebank[3:]
    splits = treebank.split('-')
    assert len(splits) == 2
    lang, corpus = splits
    lcode = lang2lcode[lang]
    short = "{}_{}".format(lcode, corpus.lower())
    return short


shorthands = []
fullnames = []
with open(list_of_tb_file) as f:
    for line in f:
        treebank = line.strip()
        fullnames.append(treebank)
        shorthands.append(treebank_to_short_name(treebank))


def filter_data(data, idx):
    data_filtered = []
    for sentence in data:
        flag = True
        for token in sentence:
            if token[idx] is None:
                flag = False
        if flag:
            data_filtered.append(sentence)
    return data_filtered


# For each treebank, we would like to find the XPOS Vocab configuration that minimizes
# the number of total classes needed to predict by all tagger classifiers. This is
# achieved by enumerating different options of separators that different treebanks might
# use, and comparing that to treating the XPOS tags as separate categories (using a
# WordVocab).
mapping = defaultdict(list)
for sh, fn in zip(shorthands, fullnames):
    print('Resolving vocab option for {}...'.format(sh))
    if not os.path.exists('data/pos/{}.train.in.conllu'.format(sh)):
        raise UserWarning('Training data for {} not found in the data directory, falling back to using WordVocab. To generate the '
                          'XPOS vocabulary for this treebank properly, please run the following command first:\n'
                          '\tbash scripts/prep_pos_data.sh {}'.format(fn, fn))
        # without the training file, there's not much we can do
        key = 'WordVocab(data, shorthand, idx=2)'
        mapping[key].append(sh)
        continue

    doc = Document(CoNLL.conll2dict(input_file='data/pos/{}.train.in.conllu'.format(sh)))
    data = doc.get([TEXT, UPOS, XPOS, FEATS], as_sentences=True)
    print(f'Original length = {len(data)}')
    data = filter_data(data, idx=2)
    print(f'Filtered length = {len(data)}')
    vocab = WordVocab(data, sh, idx=2, ignore=["_"])
    key = 'WordVocab(data, shorthand, idx=2, ignore=["_"])'
    best_size = len(vocab) - len(VOCAB_PREFIX)
    if best_size > 20:
        for sep in ['', '-', '+', '|', ',', ':']:  # separators
            vocab = XPOSVocab(data, sh, idx=2, sep=sep)
            length = sum(len(x) - len(VOCAB_PREFIX) for x in vocab._id2unit.values())
            if length < best_size:
                key = 'XPOSVocab(data, shorthand, idx=2, sep="{}")'.format(sep)
                best_size = length
    mapping[key].append(sh)

# Generate code. This takes the XPOS vocabulary classes selected above, and generates the
# actual factory class as seen in models.pos.xpos_vocab_factory.
first = True
with open(output_file, 'w') as f:
    print('''# This is the XPOS factory method generated automatically from models.pos.build_xpos_factory.
# Please don't edit it!

from models.pos.vocab import WordVocab, XPOSVocab

def xpos_vocab_factory(data, shorthand):''', file=f)

    for key in mapping:
        print("    {} shorthand in [{}]:".format('if' if first else 'elif',
                                                 ', '.join(['"{}"'.format(x) for x in mapping[key]])), file=f)
        print("        return {}".format(key), file=f)

        first = False
    print('''    else:
        raise NotImplementedError('Language shorthand "{}" not found!'.format(shorthand))''', file=f)

print('Done!')
