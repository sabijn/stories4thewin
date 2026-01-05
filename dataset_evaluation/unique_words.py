class UniqueWords():
    def __init__(self, nlp):
        self.nlp = nlp
    
    def _get_nunique_story(self, story):
        return len(set([str(token) for token in self.nlp(story) if token.pos_ != 'PUNCT']))

    def evaluate(self, text: str) -> float:
        return self._get_nunique_story(text)