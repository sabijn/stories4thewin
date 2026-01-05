import math
import pandas as pd
from collections import Counter
import re

class Grammaticality():
    L1, L2, L3 = 0.5, 0.3, 0.2       
    MIN_PROB = 1e-12

    def __init__(self, nlp, unigram_csv, bigram_csv, trigram_csv):
        self.nlp = nlp
        self.unigram, self.bigram, self.trigram, self.total_u = self._load_counts(unigram_csv, bigram_csv, trigram_csv)


    def _load_counts(self, unigram_csv, bigram_csv, trigram_csv):
        # Unigrams
        df_u = pd.read_csv(unigram_csv)
        unigram = Counter(dict(zip(df_u["pos"], df_u["freq"])))

        # Bigrams
        df_b = pd.read_csv(bigram_csv)
        bigram = Counter({(r.pos1, r.pos2): int(r.freq) for r in df_b.itertuples(index=False)})

        # Trigrams
        df_t = pd.read_csv(trigram_csv)
        trigram = Counter({(r.pos1, r.pos2, r.pos3): int(r.freq) for r in df_t.itertuples(index=False)})

        total_unigrams = sum(unigram.values())

        return unigram, bigram, trigram, total_unigrams
    
    @staticmethod
    def _frac(num: int, den: int) -> float:
        return (num / den) if den > 0 else 0.0

    def _p_unigram(self, t3: str) -> float:
        return self._frac(self.unigram.get(t3, 0), self.total_u)

    def _p_bigram(self, t2: str, t3: str) -> float:
        return self._frac(self.bigram.get((t2, t3), 0), self.unigram.get(t2, 0))

    def _p_trigram(self, t1: str, t2: str, t3: str) -> float:
        return self._frac(self.trigram.get((t1, t2, t3), 0), self.bigram.get((t1, t2), 0))

    def _p_interpolated(self, t1: str, t2: str, t3: str) -> float:
        p = self.L1 * self._p_trigram(t1, t2, t3) + self.L2 * self._p_bigram(t2, t3) + self.L3 * self._p_unigram(t3)
        return max(p, self.MIN_PROB)  # floor to avoid log(0)

    def _pos_tags(self, text: str):
        """Return list of lists of POS tags for each sentence."""
        doc = self.nlp(text)
        sents = []
        for i, sent in enumerate(doc.sents):
            tags = [t.pos_ for t in sent if not t.is_space]
            if tags:
                sents.append(tags)
        return sents
    
    def _score_sentence(self, tags):
        """
        G(sentence) = (1/n) * sum_{i=1..n} log P(K_i)
        For K_i, we use (t_{i-2}, t_{i-1}, t_i). For i < 3, we degrade context gracefully.
        """
        n = len(tags)
        if n == 0:
            return float("-inf")

        log_sum = 0.0
        for i, t3 in enumerate(tags):
            t2 = tags[i-1] if i-1 >= 0 else None
            t1 = tags[i-2] if i-2 >= 0 else None

            if t1 is None and t2 is None:
                # only unigram available
                p = self._p_unigram(t3)
            elif t1 is None:
                # bigram + unigram
                p = self.L2 * self._p_bigram(t2, t3) + self.L3 * self._p_unigram(t3)
                p = max(p, self.MIN_PROB)
            else:
                p = self._p_interpolated(t1, t2, t3)

            log_sum += math.log(p)

        return log_sum / n  # average log-prob

    def _score_story(self, text: str):
        """
        G(story) = average of G(sentence) over all sentences in the text.
        Returns (G_story, per_sentence_scores, per_sentence_tags)
        """
        sentences = self._pos_tags(text)
        if not sentences:
            return float("-inf"), [], []

        scores = [self._score_sentence(tags) for tags in sentences]
        g_story = sum(scores) / len(scores)

        return g_story, scores, sentences 

    def evaluate(self, text: str) -> float:
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)
    
        return self._score_story(text)[0]