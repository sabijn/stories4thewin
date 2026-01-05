class AvgWordLength():
    def __init__(self, nlp):
        self.nlp = nlp
    
    def _get_average_word_length(self, story):
        tokens = [len(str(w)) for w in self.nlp(story) if w.pos_ != 'PUNCT']

        return sum(tokens) / len(tokens)

    def evaluate(self, text: str) -> float:
        return self._get_average_word_length(text)