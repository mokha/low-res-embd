cd MUSE
python supervised.py --src_lang ru --tgt_lang en --src_emb ../models/aligned/rus/model.100.w2v.txt --tgt_emb ../models/aligned/en/model.100.w2v.txt --n_refinement 20 --export txt --dico_train ./data/crosslingual/dictionaries/ru-en.0-5000.txt.pos --dico_eval ./data/crosslingual/dictionaries/ru-en.5000-6500.txt.pos --emb_dim 100
#python supervised.py --src_lang fi --tgt_lang en --src_emb ../models/aligned/fin/finnish_s24_skgram_lemmas.100.w2v.txt --tgt_emb ../models/aligned/en/model.100.w2v.txt --n_refinement 20 --export txt --dico_train ./data/crosslingual/dictionaries/fi-en.0-5000.txt --dico_eval ./data/crosslingual/dictionaries/fi-en.5000-6500.txt --emb_dim 100

# better
python supervised.py --src_lang fi --tgt_lang en --src_emb ../models/aligned/fin/finnish_s24_skgram_lemmas.100.w2v.clean.txt --tgt_emb ../models/aligned/en/model.100.w2v.txt --n_refinement 20 --export txt --dico_train ./data/crosslingual/dictionaries/fi-en.0-5000.txt --dico_eval ./data/crosslingual/dictionaries/fi-en.5000-6500.txt --emb_dim 100



python supervised.py --src_lang myv --tgt_lang fi --src_emb ../models/aligned/myv-fine.txt --tgt_emb ../models/vectors-fi.txt --n_refinement 5 --export txt --dico_train ../data/myv-fin-train.txt --dico_eval ../data/myv-fin-test.txt --emb_dim 100
python supervised.py --src_lang sms --tgt_lang fi --src_emb ../models/aligned/sms-fine.txt --tgt_emb ../models/vectors-fi.txt --n_refinement 5 --export txt --dico_train ../data/sms-fin-train.txt --dico_eval ../data/sms-fin-test.txt --emb_dim 100
python supervised.py --src_lang kpv --tgt_lang ru --src_emb ../models/aligned/kpv-fine.txt --tgt_emb ../models/vectors-ru.txt --n_refinement 5 --export txt --dico_train ../data/kpv-rus-train.txt.pos --dico_eval ../data/kpv-rus-test.txt.pos --emb_dim 100
python supervised.py --src_lang mdf --tgt_lang en --src_emb ../models/aligned/mdf-fine.txt --tgt_emb ../models/vectors-en.txt --n_refinement 5 --export txt --dico_train ../data/mdf-eng-train.txt --dico_eval ../data/mdf-eng-test.txt --emb_dim 100
