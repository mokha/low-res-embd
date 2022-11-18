from gensim.models import KeyedVectors

en_emb = KeyedVectors.load_word2vec_format('./models/vectors-en.txt', binary=False)
fi_emb = KeyedVectors.load_word2vec_format('./models/vectors-fi.txt', binary=False)
ru_emb = KeyedVectors.load_word2vec_format('./models/vectors-ru.txt', binary=False)

w_vec = en_emb.word_vec('cat')
for r, similarity in en_emb.similar_by_vector(w_vec):  # , negative=['man'])
    print(f"{r}: {similarity:.4f}")
print()
for r, similarity in fi_emb.similar_by_vector(w_vec):  # , negative=['man'])
    print(f"{r}: {similarity:.4f}")