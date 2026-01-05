from itertools import combinations
from simplemma import text_lemmatizer
from scipy.stats import spearmanr
from pathlib import Path
from gensim.models import Word2Vec
import spacy
import pandas as pd
from typing import List

def pairwise_spearman(sim_lists):
    """
    sim_lists: list of 1D arrays/lists, one per model.
               e.g. [sims_model1, sims_model2, sims_model3, ...]
    
    Returns: dict with ((i, j) -> spearman_r)
    """
    results = {}
    names = list(sim_lists.keys())
    similarities = list(sim_lists.values())

    for i, j in combinations(range(len(similarities)), 2):
        r, p = spearmanr(similarities[i], similarities[j])
        results[(names[i], names[j])] = {"rho": r, "p": p}

    return results

def select_invocab_simlex(simlex: pd.DataFrame, models: Word2Vec):
    vocab_sets = [set(model.wv.key_to_index) for model in models]
    shared_vocab = set.intersection(*vocab_sets)

    in_vocab_simlex = simlex[
        simlex["lemma1"].isin(shared_vocab) &
        simlex["lemma2"].isin(shared_vocab)
    ]

    # Outputs
    word_list = list(set(in_vocab_simlex["lemma1"]).union(in_vocab_simlex["lemma2"]))
    word_pairs = list(in_vocab_simlex[["lemma1", "lemma2"]].itertuples(index=False, name=None))

    return in_vocab_simlex, word_list, word_pairs

def compute_similarity(lemma_list, model):
    pairs = lemma_list
    cos_sim = []

    for pair in pairs:
        cos_sim.append(model.wv.similarity(pair[0], pair[1]))

    return cos_sim


def lemmatize(data, nlp):
	sents = []
	for story in data:
		lines = list(nlp(story).sents)

		for sent in lines:
			words = []
			for w in sent:
				if w.is_alpha:
					try:
						lemma = text_lemmatizer(w.text.lower(), lang='nl')[0]
					except:
						lemma = w.lemma_.lower()
					words.append(lemma)
			sents.append(words)

	return sents

def train_and_save_w2v(data, language, save_name, callbacks=[]):
    if Path(save_name).exists():
        return Word2Vec.load(save_name)

    # list of sents of all stories
    if language == "nl":
        nlp = spacy.load("nl_core_news_lg") 
    elif language == "en":
        nlp = spacy.load("en_core_web_sm") 
    else:
        raise NotImplementedError
    
    sents = lemmatize(data, nlp)

    # train w2v
    model = Word2Vec(sents, 
                        min_count=5, 
                        window=5, 
                        vector_size=100, 
                        negative=5, 
                        seed=42, 
                        workers=1, 
                        sg=0, 
                        alpha=0.001, 
                        epochs=500,
                        callbacks=callbacks)
    
    model.save(save_name)

    return model
