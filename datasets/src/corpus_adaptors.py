from typing import Iterable, Optional, Iterator
from pathlib import Path
import folia.main as folia #https://foliapy.readthedocs.io/en/latest/folia.html#loading-a-document

def iter_texts_from_dataset(dataset: Iterable[str], limit_docs: Optional[int] = None) -> Iterator[str]:
    """
    Yields raw text rows from an iterable dataset (e.g., Common Crawl sample).
    """
    for i, text in enumerate(dataset):
        if limit_docs is not None and i >= limit_docs:
            break
        if text and text.strip():
            yield text

def iter_texts_from_folia_xml(dirpath: Path, limit_docs: Optional[int] = None) -> Iterator[str]:
    """
    Yields text from all FoLiA XML files in a directory (recursively).
    Requires: pip install FoLiA-tools (library name 'folia')
    """
    for i, xml in enumerate(dirpath.rglob("*.xml")):  # recursive glob
        if limit_docs is not None and i >= limit_docs:
            break
        try:
            doc = folia.Document(file=str(xml))
            txt = doc.text()
            if txt and txt.strip():
                yield txt
        except Exception as e:
            print(f"Skipping {xml}: {e}")