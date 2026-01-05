from collections import Counter
from spacy.attrs import POS, IS_SPACE
from simplemma import text_lemmatizer
from typing import Dict, Tuple

# ---------- Unigram extractor  ----------

def extract_unigrams_from_doc(doc, *, include_punct=False, include_space=False) -> Counter:
    """
    Returns Counter of (lemma, pos).
    Excludes punctuation/space by default; flip flags to include them if desired.
    """
    bag = Counter()
    for t in doc:
        if (not include_space and t.is_space) or (not include_punct and t.is_punct):
            continue
        bag[(t.lemma_, t.pos_)] += 1   # (lemma, POS)
    return bag

# ---------- Bigram extractor  ----------

def extract_bigrams_from_doc(doc) -> Counter:
    """
    Returns Counter of (lemma_{i-1}, lemma_i) from a spaCy Doc.
    """
    bag = Counter()
    # optional: skip punctuation/space tokens
    toks = [t for t in doc if not (t.is_space or t.is_punct)]
    for i in range(len(toks) - 1):
        bag[(toks[i].lemma_, toks[i+1].lemma_)] += 1
    return bag

# ---------- Linguistically constrained bigram extractor  ----------

def _collect_dep_pairs(node, bag: Counter, language: str ='nl'):
    """
    Collect linguistically constrained pairs (dependent -> head), by lemma.
    Adjust rules as you like, but keep orientation consistent with your lexicon.
    """
    # Verbal heads: subject/object relations
    if node.pos_ in ('VERB', 'AUX'):
        for child in node.children:
            if child.dep_ in ('nsubj', 'nsubj:pass', 'obj')  and (child.pos_ != "PROPN" and child.pos_ != "PRON"):
                try:
                    child_lemma = text_lemmatizer(child.text, lang=language)[0]
                except:
                    child_lemma = child.lemma_
                try:
                    node_lemma = text_lemmatizer(node.text, lang=language)[0]
                except:
                    node_lemma = node.lemma_
                bag[(child_lemma, node_lemma)] += 1

    # Nominal heads: adjectival modifiers
    if node.pos_ == 'NOUN':
        for child in node.children:
            if child.dep_ == 'amod' and child.pos_ == 'ADJ':
                try:
                    child_lemma = text_lemmatizer(child.text, lang=language)[0]
                except:
                    child_lemma = child.lemma_
                try:
                    node_lemma = text_lemmatizer(node.text, lang=language)[0]
                except:
                    node_lemma = node.lemma_
                bag[(child_lemma, node_lemma)] += 1

    for child in node.children:
        _collect_dep_pairs(child, bag)

def extract_dep_pairs_from_doc(doc) -> Counter:
    """
    Returns Counter of dependency-selected (lemma1, lemma2) pairs from a Doc.
    """
    bag = Counter()
    for sent in doc.sents:
        _collect_dep_pairs(sent.root, bag)
    return bag

# ---------- POS ngram extractor  ----------
def extract_pos_ngrams_from_doc(
    doc,
    orders: Tuple[int, ...] = (1, 2, 3),
    *,
    include_space: bool = False
) -> Dict[int, Counter]:
    """
    Returns dict: {1: Counter[(pos,)], 2: Counter[(pos1,pos2)], 3: Counter[(pos1,pos2,pos3)], ...}
    Uses Doc.to_array once for speed. Counts each sentence after dropping spaces.
    Keys are POS **IDs** (ints) for efficiency; map to strings at save-time.
    """
    out = {n: Counter() for n in orders}
    arr = doc.to_array([POS, IS_SPACE])           # shape (N, 2)
    pos_ids  = arr[:, 0]
    is_space = arr[:, 1].astype(bool)

    for sent in doc.sents:
        i0, i1 = sent.start, sent.end
        seg = pos_ids[i0:i1]
        mask = ~is_space[i0:i1] if not include_space else np.ones_like(seg, dtype=bool)
        seg = seg[mask]
        if len(seg) == 0:
            continue

        # Count n-grams via rolling zip
        for n in orders:
            if len(seg) >= n:
                # tuples of length n of POS IDs
                out[n].update(zip(*(seg[i:] for i in range(n))))
    return out






