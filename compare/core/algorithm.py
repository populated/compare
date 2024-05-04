from __future__ import annotations
from ..typing import List, Dict, Union, Tuple

import numpy as np
import spacy

from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

from ..model import (
    Responses,
    Response,
    SIMILARITY_THRESHOLD,
    TextEmbedding,
    TextPair,
    SimilarityScore
)

import sys

try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")

    print("Successfully downloaded spaCy model, please restart your program.")
    sys.exit(1)
    

def embed_text(text: str) -> np.ndarray:
    doc = nlp(text)
    return np.array([token.vector for token in doc])

def advanced_similarity(
    text1: str,
    text2: str
) -> float:
    vec1 = embed_text(text1)
    vec2 = embed_text(text2)

    _, _, v1 = np.linalg.svd(vec1)
    _, _, v2 = np.linalg.svd(vec2)

    cos_sim = np.dot(v1[0], v2[0]) / \
        (np.linalg.norm(v1[0]) * np.linalg.norm(v2[0]))
    return cos_sim

class TextComparer:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self._calculate_embeddings()

    def _calculate_embeddings(self: "TextComparer") -> List[TextEmbedding]:
        return self.vectorizer.fit_transform(self.texts).toarray()

    def compare_texts(
        self: "TextComparer",
        input_text: str,
        advanced: bool = False
    ) -> Dict[str, Union[Response, SimilarityScore]]:
        input_embedding = self.vectorizer.transform([input_text]).toarray()[0]
        similarities = [
            (text, advanced_similarity(input_text, text) if advanced else 1 - cosine(input_embedding, text_embedding))
            for text, text_embedding in zip(self.texts, self.embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        closest, score = similarities[0]
        similar = [text for text, score in similarities[1:] if score >= SIMILARITY_THRESHOLD]

        return {
            "closest": closest,
            "score": score,
            "similar": similar
        }

__all__ = [
    "embed_text",
    "advanced_similarity",
    "TextComparer"
]
