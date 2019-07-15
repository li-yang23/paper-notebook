import logging
from io import open
from os import path
from time import time
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from six.moves import zip

from . import graph

logger = logging.getLogger("deepwalk")

__current_graph = None

__vertex2str = None

def count_words(file):
    """Counts the word frequences in a list of sentences.

    Note:
    This is a helper function for parallel execution of 'Vocabulary.from_text'
    method
    """
    c = Counter()
    with open(file,'r') as f:
        for l in f:
            words = l.strip().split()
            c.update(words)
    return c

def count_textfiles(files,workers=1):
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words,files):
            c.update(c_)
    return c

