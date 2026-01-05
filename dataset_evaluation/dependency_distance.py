import re

class DependencyDistance():
	def __init__(self, nlp):
		self.nlp = nlp

	def _dep_dist(self, text):
		"""
		This is a custom implementation of DD based on the textdescriptives implementation of TD
		Yet, we want to split on our own utterance markers, not on spaCy's built-in senter
		We found out that it doesn't work optimal
		So here we calculate dependency distance according to the implementation of Liu (2008)
		"""
		text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
		text = re.sub(' +', ' ', text)

		doc = self.nlp(text)
		sent_list = list(doc.sents)

		DD = []

		for sent in sent_list:
			for token in sent:
				if token.dep_ != "ROOT": 
					DD.append(abs(token.head.i - token.i))
		
		DD = [distance for distance in DD if distance != 0] #-- drop DDs for whitespace, which are zero, and not included in w
		
		return (1 / (len(doc) - len(sent_list))) * sum(DD)

	def evaluate(self, text: str) -> float:
		return self._dep_dist(text)