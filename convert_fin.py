import lwvlib
import io
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
# read the finnish embeddings
wv = lwvlib.load("./models/aligned/fin/finnish_s24_skgram_lemmas.bin", 10000, 500000)

# write them as text
# wv.words
with io.open('./models/aligned/fin/finnish_s24_skgram_lemmas.txt', 'w', encoding='utf-8') as outfile:
    lines = []
    for w in wv.words:
        lines.append(w + ' ' + ' '.join(wv.vec(w).astype('str')) + "\n")
    lines[-1] = lines[-1][:-1]
    outfile.writelines(lines)
