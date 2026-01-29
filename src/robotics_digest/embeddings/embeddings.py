# robotics_digest/embeddings.py
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
