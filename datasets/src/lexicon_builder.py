import pandas as pd
import numpy as np

from pathlib import Path
from collections import Counter
from tqdm.notebook import tqdm

from typing import Iterable, Callable, Optional, Sequence, Tuple, Dict, Any

from .extractors import extract_pos_ngrams_from_doc

# ---------- Build POS lexicon  ----------
def build_pos_ngram_lexicons(
    texts: Iterable[str],
    nlp,
    *,
    orders: Tuple[int, ...] = (1, 2, 3),
    batch_size: int = 100,
    n_process: int = 1,
    limit_docs: Optional[int] = None,
) -> Dict[int, Counter]:
    """
    Streams texts and aggregates POS n-grams for all requested orders in ONE pass.
    """
    agg = {n: Counter() for n in orders}
    doc_count = 0
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), desc="Building POS n-grams"):
        if limit_docs is not None and doc_count >= limit_docs:
            break
        bags = extract_pos_ngrams_from_doc(doc, orders=orders)
        for n in orders:
            agg[n].update(bags[n])
        doc_count += 1
    return agg

# ---- helper: load a single POS n-gram CSV with schema checks ----
def _load_pos_ngram_df(path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = (["pos", "freq"] if n == 1
                else [*(f"pos{i}" for i in range(1, n+1)), "freq"])
    if set(df.columns) != set(expected):
        raise ValueError(
            f"Unexpected columns in {path.name}: {list(df.columns)}; expected {expected}"
        )
    # reorder columns if needed
    df = df[expected]
    return df

# ---- main wrapper: load-or-build for POS n-grams ----
def load_or_build_pos_ngram_lexicons(
    out_paths: Dict[int, Path],                 # e.g. {1: uni.csv, 2: bi.csv, 3: tri.csv}
    texts_factory: Callable[[], Iterable[str]], # returns a fresh iterable of texts
    nlp,
    *,
    orders: Tuple[int, ...] = (1, 2, 3),
    batch_size: int = 100,
    n_process: int = 1,
    limit_docs: Optional[int] = None,
) -> Dict[int, pd.DataFrame]:
    """
    If all requested CSVs exist, load and return them.
    Otherwise, build only the missing orders in a single pass and save, then load all.
    Returns a dict {n: DataFrame}.
    """
    # Which n-grams are missing?
    missing = tuple(n for n in orders if not out_paths.get(n, None) or not out_paths[n].exists())

    if missing:
        # Build only the missing ones in ONE pass over the corpus
        counters = build_pos_ngram_lexicons(
            texts=texts_factory(),
            nlp=nlp,
            orders=missing,
            batch_size=batch_size,
            n_process=n_process,
            limit_docs=limit_docs,
        )
        # Save newly built ones (maps POS IDs -> strings at write time)
        save_pos_ngram_lexicons(
            counters=counters,
            out_paths={n: out_paths[n] for n in missing},
            nlp=nlp,
        )

    # Load all requested orders from disk
    return {n: _load_pos_ngram_df(out_paths[n], n) for n in orders}

# ---------- Build regular lexicon  ----------

def build_lexicon(
    texts: Iterable[str],
    nlp,
    extractor: Callable,                 # one of the extract_*_from_doc functions
    *,
    batch_size: int = 100,
    n_process: int = 1,
    flush_every: int = 100_000,          
    cache_path: Optional[Path] = None, 
    key_names: Sequence[str] = ("lemma1","lemma2")
) -> Counter:
    """
    Streams texts through spaCy, applies 'extractor' to each Doc, and aggregates a Counter of (lemma1, lemma2) -> freq.
    """
    agg = Counter()
    seen_since_flush = 0

    # Faster: use nlp.pipe to batch & parallelize
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), desc="Building lexicon"):
        bag = extractor(doc)
        agg.update(bag)
        seen_since_flush += sum(bag.values())

        if cache_path and seen_since_flush >= flush_every:
            # dump a partial (append-mode parquet/csv could be used; here we overwrite a temp snapshot)
            _save_counter_as_csv(agg, cache_path)
            seen_since_flush = 0

    return agg

# ---------- Save lexicons ----------
def _save_counter_as_csv(
    counter: Counter,
    path: Path,
    key_names: Sequence[str] = ("lemma1", "lemma2"),
    key_transform: Optional[Callable[[Tuple[Any, ...]], Tuple[Any, ...]]] = None,
) -> pd.DataFrame:
    """
    Save Counter with tuple keys to CSV columns [*key_names, 'freq'].
    key_transform lets you map keys before writing (e.g., POS id -> tag string).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    expected_len = len(key_names)
    for k, v in counter.items():
        if not isinstance(k, tuple):
            k = (k,)
        if key_transform:
            k = key_transform(k)
        if len(k) != expected_len:
            raise ValueError(f"Key length {len(k)} != len(key_names) {expected_len}. Key={k}")
        rows.append((*k, v))

    df = pd.DataFrame(rows, columns=[*key_names, "freq"])
    if not df.empty:
        df["freq"] = df["freq"].astype(np.int64)
    df.to_csv(path, index=False)
    return df


def save_lexicon(counter: Counter, out_path: Path, key_names: Sequence[str]) -> pd.DataFrame:
    _save_counter_as_csv(counter, out_path, key_names)
    return pd.read_csv(out_path)


def load_or_build_lexicon(
    out_path: Path,
    builder_fn: Callable[[], Counter],
    key_names: Sequence[str] = ("lemma1","lemma2"),
) -> pd.DataFrame:
    """
    If 'out_path' exists, load CSV; otherwise call builder_fn(), save, and return the DataFrame.
    """
    if out_path.exists():
        df = pd.read_csv(out_path)

        expected = {*key_names, "freq"}
        if set(df.columns) != expected:
            raise Exception(f'The old csv has not the correct columns. Check if you are reading in the right csv.')
        return df
    else:
        counter = builder_fn()
        return save_lexicon(counter, out_path, key_names=key_names)


def save_pos_ngram_lexicons(
    counters: Dict[int, Counter],
    out_paths: Dict[int, Path],
    nlp,
) -> Dict[int, pd.DataFrame]:
    """
    Writes each n-gram Counter to CSV with schema:
      n=1: ['pos','freq']
      n=2: ['pos1','pos2','freq']
      n=3: ['pos1','pos2','pos3','freq']
    Returns DataFrames that were written.
    """
    id2str = nlp.vocab.strings

    def key_names_for(n: int) -> Tuple[str, ...]:
        return ("pos",) if n == 1 else tuple(f"pos{i}" for i in range(1, n+1))

    def key_transform_for(n: int) -> Callable[[Tuple[int, ...]], Tuple[str, ...]]:
        return lambda key: tuple(id2str[k] for k in key)

    dfs = {}
    for n, counter in counters.items():
        names = key_names_for(n)
        kxf   = key_transform_for(n)
        df = _save_counter_as_csv(counter, out_paths[n], key_names=names, key_transform=kxf)
        dfs[n] = df
    return dfs