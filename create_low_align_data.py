import io
from mikatools import *
import numpy as np
from gensim.models import KeyedVectors
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split

supported_langs = {'eng', 'rus', 'fin'}

low_res_langs = {
    'myv': './UD/UD_Erzya-JR/myv_jr-ud-test.conllu',
    'kpv': './UD/UD_Komi_Zyrian-Lattice/kpv_lattice-ud-test.conllu',
    'sms': './UD_Skolt_Sami-Giellagas/sms_giellagas-ud-test.conllu',
    'mdf': './UD/UD_Moksha-JR/mdf_jr-ud-test.conllu',
}


for lang, ud_path in low_res_langs.items():
    d = json_load("./link-pred/dicts/translations_{}.json".format(lang))

    w2v = KeyedVectors.load_word2vec_format('./models/aligned/{}-fine.txt'.format(lang), binary=False)

    counts = defaultdict(int)
    lang_pairs = defaultdict(list)

    for lemma, mg in d.items():
        lemma = lemma.strip()
        if lemma not in w2v.vocab:
            continue

        for m in mg.values():
            for tgt, translations in m.items():
                if tgt in supported_langs:
                    counts[tgt] += 1

                    for t in translations:
                        if ' ' not in t:
                            lang_pairs[tgt].append((lemma, t.strip()))

    print(lang, ':')
    print(counts)
    print()
    for k in lang_pairs:
        print(k, len(lang_pairs[k]))

        data = lang_pairs[k]

        training_dataset, test_dataset = train_test_split(data, test_size=0.2)

        with io.open('./data/{}-{}-train.txt'.format(lang, k), 'w', encoding='utf-8') as of:
            training_dataset = [' '.join(t).lower() for t in training_dataset]
            of.write('\n'.join(training_dataset))

        with io.open('./data/{}-{}-test.txt'.format(lang, k), 'w', encoding='utf-8') as of:
            test_dataset = [' '.join(t).lower() for t in test_dataset]
            of.write('\n'.join(test_dataset))

files = ['./data/{}-{}-train.txt'.format('kpv', 'rus'), './data/{}-{}-test.txt'.format('kpv', 'rus')]

w2v = KeyedVectors.load_word2vec_format('./models/aligned/rus/model.txt', binary=False)
l2p = defaultdict(list)
for w in w2v.vocab:
    lemma, pos = w.split('_')
    l2p[lemma].append(pos)

for f in files:
    with io.open(f, 'r', encoding='utf-8') as f1, io.open(f + '.pos', 'w', encoding='utf-8') as f2:
        for _l in f1:
            kpv, rus = _l.rstrip().split()
            if rus in l2p:
                for pos in l2p[rus]:
                    f2.write(f"{kpv} {rus}_{pos}\n".lower())
