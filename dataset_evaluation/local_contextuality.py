# local_contextuality_component.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class LocalContextuality:
    """
    Computes local contextuality as the average cosine similarity between
    consecutive sentences in a text.

    Parameters
    ----------
    nlp : spaCy Language
        Used for sentence segmentation.
    model_path : str
        Path or HuggingFace model ID for sentence embeddings.
        Default: "bert-base-uncased".
    min_items : int
        Minimum number of sentences required to compute a score. If fewer, returns 0.0.
    """

    def __init__(self, nlp, model_path: str = "bert-base-uncased", min_items: int = 2):
        self.nlp = nlp
        self.model_path = model_path
        self.model = SentenceTransformer(model_path)
        self.min_items = min_items

    def _calculate_lc(self, sents):
        if len(list(sents)) < self.min_items:
            return 0.0

        try:
            # Encode all sentences
            X = self.model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)

            # Cosine similarity of consecutive pairs
            sims = []
            for i in range(len(X) - 1):
                sims.append(float(np.dot(X[i], X[i + 1])))

            return float(np.mean(sims)) if sims else 0.0

        except Exception:
            return 0.0

    def evaluate(self, text: str) -> float:
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)

        sents = list(map(str, list(self.nlp(text).sents)))

        return self._calculate_lc(sents)
