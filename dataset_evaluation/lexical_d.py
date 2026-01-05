from nltk.tokenize import word_tokenize
from lexical_diversity import lex_div as ld

class LexicalDiversity():
    def __init__(self):
        pass
    
    def _msTTR(self, text):
        """
        Mean segmental TTR: divides the text in windows of the specified size and calculates the average TTR over them
        """
        return ld.msttr(word_tokenize(text), window_length=10)

    def _maTTR(self, text):
        """
        Moving average TTR: (MATTR). TTR for a moving window from first to last token, and calculate the average TTR over them
        """
        return ld.mattr(word_tokenize(text), window_length=10)

    def _mtld(self, text):
        """
        MTLD: the average number of words for which a consecutive TTR is maintained.
        """
        return ld.mtld(word_tokenize(text))

    def _moving_mtld(self, text):
        """
        Moving MTLD
        """
        return ld.mtld_ma_wrap(word_tokenize(text))

    def _moving_mtld_bi(self, text):
        """
        MTLD bi-directional
        """
        return ld.mtld_ma_bid(word_tokenize(text))

    def _HDD(self, text):
        """
        HDD (https://journals.sagepub.com/doi/pdf/10.1177/0265532207080767)
        """
        return ld.hdd(word_tokenize(text))

    def evaluate(self, text: str, methods="moving_mtld") -> dict:
        results = {}

        # Mapping method names to internal functions
        method_map = {
            "msTTR": self._msTTR,
            "maTTR": self._maTTR,
            "mtld": self._mtld,
            "moving_mtld": self._moving_mtld,
            "moving_mtld_bi": self._moving_mtld_bi,
            "HDD": self._HDD,
        }

        # Determine which methods to run
        if isinstance(methods, str):
            methods = [methods]

        if methods[0] == "all":
            selected = method_map.keys()
            
        elif isinstance(methods, (list, tuple)):
            selected = methods
        else:
            raise ValueError("Invalid value for 'methods'.")

        # Run selected methods
        for name in selected:
            if name in method_map:
                results[name] = method_map[name](text)
            else:
                raise ValueError(f"Method '{name}' not found in evaluator.")

        return results
