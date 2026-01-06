import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np
from simplemma import text_lemmatizer

class CreativePerplexity():
    def __init__(self, nlp, language, mode, unigram, bigram, constrained_bigram):
        self.nlp = nlp
        self.mode = mode
        self.language = language
        self.unigram = self._load_reference_corpora(unigram)
        self.bigram = self._load_reference_corpora(bigram)
        self.constrained_bigram = self._load_reference_corpora(constrained_bigram)

    def _load_reference_corpora(self, data_path: str) -> pd.DataFrame:
        if Path(data_path).exists():
            return pd.read_csv(data_path)
        
        raise Exception(f'Reference corpora are not registered.')

    def _collect_dep_pairs(self, node, bag: Counter):
        """
        Collect linguistically constrained pairs.
        Here: (child -> head) for selected relations.
        Adjust the rules to your needs, but keep orientation consistent with lexicon.
        """
        # Verbal heads: subject/object relations
        if node.pos_ in ('VERB', 'AUX'):
            for child in node.children:
                if child.dep_ in ('nsubj', 'nsubj:pass', 'obj') and child.text.isalnum():
                    child_lemma = text_lemmatizer(child.text, lang=self.language)[0]
                    node_lemma = text_lemmatizer(node.text, lang=self.language)[0]
                    bag[(child_lemma, node_lemma)] += 1

        # Nominal heads: adjectival modifiers
        if node.pos_ == 'NOUN' and node.text.isalnum():
            for child in node.children:
                if child.dep_ == 'amod' and child.pos_ == 'ADJ' and child.text.isalnum():
                    child_lemma = text_lemmatizer(child.text, lang=self.language)[0]
                    node_lemma = text_lemmatizer(node.text, lang=self.language)[0]
                    bag[(child_lemma, node_lemma)] += 1

        # Recurse
        for child in node.children:
            self._collect_dep_pairs(child, bag)

    def _extract_dep_pairs(self, story):
        """
        Return Counter of linguistically constrained (lemma1, lemma2) pairs.
        Current orientation: (dependent lemma, head lemma).
        """
        bag = Counter()
        for sent in self.nlp(story).sents:
            self._collect_dep_pairs(sent.root, bag)
        return bag

    def _compute_perplexity_from_counts(
        self,
        story_counts: Counter,
        lex: pd.DataFrame,       
        *,
        smoothing: str = "kneser_ney",   
        alpha: float = 0.1,              
        unk_token: str = "<unk>",
        return_frame: bool = False
    ):
        
        # Fixed vocab for lemma2 (after cutoff)
        vocab2 = set(lex['lemma2'].unique())
        global_V = len(vocab2)

        # ---------- 2) Prepare story pairs and map OOV lemma2 → <unk> ----------
        if not story_counts:
            return (None, pd.DataFrame(columns=['lemma1','lemma2','count'])) if return_frame else None

        def map_l2(l2):
            return l2 if l2 in vocab2 else unk_token

        story_pairs = [ (l1, map_l2(l2), c) for (l1, l2), c in story_counts.items() ]
        story_df = pd.DataFrame(story_pairs, columns=['lemma1','lemma2','count'])

        # Merge training counts for these pairs
        story_df = story_df.merge(lex, how='left', on=['lemma1','lemma2'])
        story_df['freq'] = story_df['freq'].fillna(0).astype(int)

        # Totals per history
        hist_totals = lex.groupby('lemma1', as_index=False)['freq'].sum().rename(columns={'freq':'hist_total'})
        story_df = story_df.merge(hist_totals, how='left', on='lemma1')
        story_df['hist_total'] = story_df['hist_total'].fillna(0).astype(int)

        # ---------- 3A) Kneser–Ney smoothing ----------
        cont_df = (
            lex[lex['freq'] > 0]
            .groupby('lemma2', as_index=False)['lemma1']
            .nunique()
            .rename(columns={'lemma1':'cont_count'})
        )
        cont_df = pd.DataFrame({'lemma2': list(vocab2)}).merge(cont_df, how='left', on='lemma2')
        cont_df['cont_count'] = cont_df['cont_count'].fillna(0).astype(int)

        total_types = int((lex['freq'] > 0).sum())
        cont_df['p_cont'] = np.where(
            total_types > 0,
            cont_df['cont_count'] / total_types,
            1.0 / global_V
        )

        story_df = story_df.merge(cont_df[['lemma2','p_cont']], how='left', on='lemma2')
        story_df['p_cont'] = story_df['p_cont'].fillna(1.0 / global_V)

        # Distinct followers per history
        T_df = (
            lex[lex['freq'] > 0]
            .groupby('lemma1', as_index=False)['lemma2']
            .nunique()
            .rename(columns={'lemma2':'T_followers'})
        )
        story_df = story_df.merge(T_df, how='left', on='lemma1')
        story_df['T_followers'] = story_df['T_followers'].fillna(0).astype(int)

        # Discount D
        counts = lex['freq'].values
        N1 = int(np.sum(counts == 1))
        N2 = int(np.sum(counts == 2))
        D = (N1 / (N1 + 2 * N2)) if (N1 + 2 * N2) > 0 else 0.75

        hist = story_df["hist_total"].replace(0, np.nan)  # avoid division by zero
        lam = (D * story_df["T_followers"]) / hist
        lam = lam.fillna(1.0)  # unseen history → full backoff

        hist_safe = story_df['hist_total'].replace(0, np.nan)
        first_term = np.maximum(story_df['freq'] - D, 0.0) / hist_safe
        first_term = first_term.fillna(0.0)

        probs = first_term + lam * story_df['p_cont']
        probs = np.maximum(probs, np.finfo(float).tiny)

        diag_cols = ['lemma1','lemma2','count','freq','hist_total','T_followers','p_cont']
        story_df_out = story_df[diag_cols].copy()
        story_df_out['prob'] = probs

        # ---------- 4) Compute perplexity ----------
        N = int(story_df['count'].sum())
        if N == 0:
            return (None, story_df_out) if return_frame else None

        avg_logp = (np.log(probs) * story_df['count']).sum() / N
        ppl = float(np.exp(-avg_logp))

        if return_frame:
            return ppl, story_df_out
        return ppl

    def _perplexity_for_creativity(
        self,
        story: str,
        *,
        alpha: float,
        vocab: str,
        with_pos_unigram: bool, 
        unigram_cols,        # or ("lemma","pos") to match lexicon
        return_frame: bool,
    ):
        if self.mode == "bigram":
            raise NotImplementedError
        elif self.mode == "dep":
            counts = self._extract_dep_pairs(story)
            return self._compute_perplexity_from_counts(
                counts, self.constrained_bigram, alpha=alpha, return_frame=return_frame
            )
        elif self.mode == "unigram":
            raise NotImplementedError
        else:
            raise ValueError("mode must be 'unigram', 'bigram', or 'dep'")
    
    def evaluate(self, text: str, vocab="per_history", alpha=1.0, with_pos_unigram=True,
                 unigram_cols=("lemma","pos"), return_frame=False
                 ) -> float:
        return self._perplexity_for_creativity(text,
                                               vocab=vocab,
                                               alpha=alpha,
                                               with_pos_unigram=with_pos_unigram,
                                               unigram_cols=unigram_cols,
                                               return_frame=return_frame)