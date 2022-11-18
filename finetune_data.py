import io
from uralicNLP import uralicApi
from mikatools import *
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from collections import defaultdict
from uralicNLP.ud_tools import UD_collection
import numpy as np
import gensim

low_res_langs = {
    'myv': './UD/UD_Erzya-JR/myv_jr-ud-test.conllu',
    'kpv': './UD/UD_Komi_Zyrian-Lattice/kpv_lattice-ud-test.conllu',
    'sms': './UD/UD_Skolt_Sami-Giellagas/sms_giellagas-ud-test.conllu',
    'mdf': './UD/UD_Moksha-JR/mdf_jr-ud-test.conllu',
}

# WORD2VEC
W2V_SIZE = 100
W2V_WINDOW = 5
W2V_EPOCH = 100
W2V_MIN_COUNT = 2

for lang, ud_path in low_res_langs.items():
    w2v_path = './models/aligned/{}.txt'.format(lang)

    dicts = []
    with io.open(w2v_path, 'r', encoding='utf-8') as ff:
        for _l in ff.readlines()[1:]:
            _l = _l.rstrip()
            dicts.append(_l.split(' ')[0])

    ud = UD_collection(io.open(ud_path, encoding="utf-8"))

    # write the data (if needed)
    sentences = [[word.lemma for word in sentence if word.lemma in dicts] for sentence in ud]
    docs = [' '.join([word.lemma for word in sentence if word.lemma in dicts]) + "\n" for sentence in ud]
    docs = [_l for _l in docs if _l != "\n" and _l.count(' ') > 0]
    with io.open('./data/{}.txt'.format(lang), 'w', encoding='utf-8') as sent_file:
        sent_file.writelines(docs)

    # finetune
    model_2 = Word2Vec(size=100, min_count=1)
    model_2.build_vocab(sentences)
    total_examples = model_2.corpus_count
    model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    model_2.build_vocab([list(model.vocab.keys())], update=True)
    model_2.intersect_word2vec_format(w2v_path, binary=False, lockf=1.0)
    model_2.train(sentences, total_examples=total_examples, epochs=model_2.iter)
    word_vectors = model_2.wv
    word_vectors.save_word2vec_format('./models/aligned/{}-fine.txt'.format(lang), binary=False)
