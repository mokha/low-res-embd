import io
from uralicNLP import uralicApi
from mikatools import *
from gensim.models import KeyedVectors
from collections import defaultdict
from uralicNLP.ud_tools import UD_collection
import numpy as np




def centroid(vectors):
    if vectors:
        return np.mean(vectors, axis=0)
    return None


en_emb = KeyedVectors.load_word2vec_format('./models/vectors-en.txt', binary=False)
fi_emb = KeyedVectors.load_word2vec_format('./models/vectors-fi.txt', binary=False)
ru_emb = KeyedVectors.load_word2vec_format('./models/vectors-ru.txt', binary=False)

rus_vocab = {}
for w in ru_emb.vocab:
    lemma, pos = w.split('_')
    rus_vocab[lemma] = ru_emb.get_vector(w)

fin_vocab = {}
for w in fi_emb.vocab:
    _w = w.replace('#', '')
    fin_vocab[_w] = fi_emb.get_vector(w)

eng_vocab = {}
for w in en_emb.vocab:
    eng_vocab[w] = en_emb.get_vector(w)

supported_langs = {
    'eng': (en_emb, eng_vocab),
    'fin': (fi_emb, fin_vocab),
    'rus': (ru_emb, rus_vocab),
}

low_res_langs = {
    'myv': './UD/UD_Erzya-JR/myv_jr-ud-test.conllu',
    'kpv': './UD/UD_Komi_Zyrian-Lattice/kpv_lattice-ud-test.conllu',
    'sms': './UD_Skolt_Sami-Giellagas/sms_giellagas-ud-test.conllu',
    'mdf': './UD/UD_Moksha-JR/mdf_jr-ud-test.conllu',
}

np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

for lang, ud_path in low_res_langs.items():
    d = json_load("./link-pred/dicts/translations_{}.json".format(lang))

    wv = {}
    for lemma, mg in d.items():
        if ' ' in lemma:
            continue
        lemma = lemma.strip()
        v = []
        for m in mg.values():
            for tgt, translations in m.items():
                if tgt in supported_langs:
                    for t in translations:
                        if t in supported_langs[tgt][1]:
                            v.append(supported_langs[tgt][1][t])
        if v:
            wv[lemma] = centroid(v)

    with io.open('./models/aligned/{}.txt'.format(lang), 'w', encoding='utf-8') as vec_file:
        vec_file.write(f"{len(wv)} 100\n")
        lines = []
        for w, v in wv.items():
            lines.append(w + ' ' + ' '.join(v.astype('str')) + "\n")
        lines[-1] = lines[-1][:-1]
        vec_file.writelines(lines)


