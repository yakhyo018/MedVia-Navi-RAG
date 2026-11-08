"""
Text → vectors. Two implementations behind one Protocol:

- E5Embedder — intfloat/multilingual-e5-small (MIT licence, 384 dimensions, ~100 languages).
  E5 was trained with role prefixes: a document is "passage: …", a question is "query: …".
  Without them retrieval quality drops noticeably — the model card says so, and it is the
  most common E5 mistake.
- HashEmbedder — deterministic bag-of-words hashing, no model, no torch. For tests and CI only:
  it puts texts that share words close together, which is all a test needs.
"""

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small', device: str = 'cpu'):
        # Imported here: torch is the [ml] extra, and CI never builds this class.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 384)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True: unit vectors, so cosine similarity = dot product.
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f'passage: {t}' for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f'query: {text}'])[0]


_WORD = re.compile(r'\w+', re.UNICODE)


class HashEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
