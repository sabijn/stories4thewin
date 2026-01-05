import numpy as np
import re

class SyntacticDepth():
    def __init__(self, nlp):
        self.nlp = nlp

    def _walk_tree(self, node, depth):
        """
        Traverse tree
        """
        if node.n_lefts + node.n_rights > 0:
            return max(self._walk_tree(child, depth + 1) for child in node.children)
        else:
            return depth

    # sum and average found treedepths per sentence
    def _average_treedepth(self, raw_story):
        """
        Sum and average treedepths per sentence
        """
        treedepths = []
        sents = list(self.nlp(raw_story).sents)

        for sent in sents:
            if sent == '':
                continue
            treedepths.append(self._walk_tree(sent.root, 0))

        return np.sum(treedepths) / len(treedepths)
    
    def evaluate(self, text):
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)

        return self._average_treedepth(text)