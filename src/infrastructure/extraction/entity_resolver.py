import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from src.domain.interfaces import IEntityResolverService

class EntityResolver(IEntityResolverService):
    def __init__(self, similarity_threshold: float = 0.85):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold
        self.canonical_map: Dict[str, str] = {} # raw_name -> canonical_name
        self.canonical_embeddings: Dict[str, np.ndarray] = {}
        self.aliases: Dict[str, List[str]] = {} # canonical_name -> list of aliases

    def resolve(self, raw_name: str) -> str:
        normalized = raw_name.strip()
        if not normalized:
            return "UNKNOWN"
            
        if normalized in self.canonical_map:
            return self.canonical_map[normalized]

        vec = self.model.encode(normalized)

        best_match = None
        highest_sim = -1.0

        for canon_name, canon_vec in self.canonical_embeddings.items():
            norm_product = (np.linalg.norm(vec) * np.linalg.norm(canon_vec))
            if norm_product == 0:
                sim = 0.0
            else:
                sim = float(np.dot(vec, canon_vec) / norm_product)

            if sim > highest_sim:
                highest_sim = sim
                best_match = canon_name

        if best_match and highest_sim >= self.threshold:
            self.canonical_map[normalized] = best_match
            if normalized not in self.aliases[best_match]:
                self.aliases[best_match].append(normalized)
            return best_match
        else:
            self.canonical_map[normalized] = normalized
            self.canonical_embeddings[normalized] = vec
            self.aliases[normalized] = [normalized]
            return normalized

    def get_aliases(self, canonical_name: str) -> List[str]:
        return self.aliases.get(canonical_name, [canonical_name])
