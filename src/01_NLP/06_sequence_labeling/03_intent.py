import pandas as pd
import numpy as np

from dataclasses import dataclass

@dataclass(frozen=True)
class Embeddings:
    word_to_id: dict[str, int]
    id_to_word: dict[int, str]
    embeddings: np.ndarray


