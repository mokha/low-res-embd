import io
from gensim.models import KeyedVectors
from collections import defaultdict

w2v = KeyedVectors.load_word2vec_format('./models/aligned/rus/model.txt', binary=False)

files = ['./MUSE/data/crosslingual/dictionaries/ru-en.0-5000.txt',
         './MUSE/data/crosslingual/dictionaries/ru-en.5000-6500.txt']

l2p = defaultdict(list)
for w in w2v.vocab:
    lemma, pos = w.split('_')
    l2p[lemma].append(pos)

for f in files:
    with io.open(f, 'r', encoding='utf-8') as f1, io.open(f + '.pos', 'w', encoding='utf-8') as f2:
        for _l in f1:
            rus, eng = _l.rstrip().split()
            if rus in l2p:
                for pos in l2p[rus]:
                    f2.write(f"{rus}_{pos} {eng}\n".lower())
