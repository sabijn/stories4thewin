import re

class AverageComponents():
    def __init__(self, nlp):
        self.nlp = nlp
    
    @staticmethod
    def _safe_divide(numerator, denominator) -> float:
        try:
            index = numerator/denominator
        except:
            index = 0

        return index

    def _extract_complements(self, story: str) -> float:
        """
        Function to average (open) clausal complements
        Returns average (float)
        """
        comps = 0

        doc = self.nlp(story)
        sents = len(list(doc.sents))

        for token in doc:
            if token.dep_ == 'ccomp':
                # clausal complement ('I think that he is lying')
                comps += 1
            elif token.dep_ == 'xcomp':
                # open clausal complement ('He made her cry')
                comps += 1
        
        return self._safe_divide(comps, sents)

    def evaluate(self, text: str) -> float:
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)

        return self._extract_complements(text)