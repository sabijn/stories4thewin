

# vendi_score_component.py
from typing import List, Sequence
import nltk
from nltk.translate.bleu_score import SmoothingFunction
import re

class SelfBleuScore:
    def __init__(
        self,
        nlp,
        min_items: int = 2,
    ):
        self.nlp = nlp
        self.min_items = int(min_items)

    def get_bleu_score(self, references, hypothesis):
        ngram = 3
        weight = tuple((1. / ngram for _ in range(ngram)))

        score = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weight,
                                                                        smoothing_function=SmoothingFunction().method1)
        return score

    def get_sentences(self, story):
        tokenized_sentences = []

        for sent in self.nlp(story).sents:
            tokenized_sentences.append([str(w) for w in sent if w.pos_ != 'PUNCT'])

        return tokenized_sentences


    def calculate_self_bleu(self, sentences):
        all_scores = []

        for idx, hyp_sent in enumerate(sentences):
            references = sentences.copy()
            references.pop(idx)
                
            bleu_score = self.get_bleu_score(references, hyp_sent)
            all_scores.append(bleu_score)
        
        return sum(all_scores) / len(all_scores)
    

    def evaluate(self, text: str) -> float:
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)
    
        sents = self.get_sentences(text)
        if len(sents) < self.min_items:
            return 0.0

        try:
            return self.calculate_self_bleu(sents)
        except Exception:
            return 0.0
