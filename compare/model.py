from __future__ import annotations

from .typing import Union, List, Dict, Tuple
import numpy as np

Response = Dict[str, Union[str, float]]
Responses = List[str]
SIMILARITY_THRESHOLD = 0.6

TextEmbedding = np.ndarray
TextPair = Tuple[str, str]
SimilarityScore = float
