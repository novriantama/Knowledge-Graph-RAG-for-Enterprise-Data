import re
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from src.domain.interfaces import IEntityResolverService

class EntityResolver(IEntityResolverService):
    """Production Entity Resolution Engine.
    
    Collapses entity variants (e.g. 'Acme Corp', 'Acme Corporation', 'ACME') into one canonical graph node.
    Combines string normalization (suffix removal, lowercasing) with embedding cosine similarity thresholding,
    and maintains a persistent alias list for Neo4j node attributes.
    """

    CORPORATE_SUFFIXES = re.compile(
        r'\b(corp|corporation|inc|incorporated|llc|limited|gmbh|co|ltd|systems|platform|service|microservice)\b',
        re.IGNORECASE
    )

    def __init__(self, similarity_threshold: float = 0.80):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold
        self.canonical_map: Dict[str, str] = {}           # raw_name -> canonical_name
        self.normalized_map: Dict[str, str] = {}          # normalized_stem -> canonical_name
        self.canonical_embeddings: Dict[str, np.ndarray] = {}
        self.aliases: Dict[str, List[str]] = {}           # canonical_name -> list of aliases

    def _normalize(self, name: str) -> str:
        """Strips punctuation, legal suffixes, and converts to lowercase stem."""
        cleaned = self.CORPORATE_SUFFIXES.sub('', name)
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        return cleaned.strip().lower()

    def resolve(self, raw_name: str) -> str:
        raw_trimmed = raw_name.strip()
        if not raw_trimmed:
            return "UNKNOWN"

        # 1. Exact match on raw name
        if raw_trimmed in self.canonical_map:
            return self.canonical_map[raw_trimmed]

        norm_stem = self._normalize(raw_trimmed)

        # 2. Match on normalized stem (collapses "Acme Corp" and "Acme Corporation")
        if norm_stem and norm_stem in self.normalized_map:
            canonical = self.normalized_map[norm_stem]
            self.canonical_map[raw_trimmed] = canonical
            if raw_trimmed not in self.aliases[canonical]:
                self.aliases[canonical].append(raw_trimmed)
            return canonical

        # 3. Embedding Similarity Match across existing canonical nodes
        vec = self.model.encode(raw_trimmed)
        best_match = None
        highest_sim = -1.0

        for canon_name, canon_vec in self.canonical_embeddings.items():
            norm_product = (np.linalg.norm(vec) * np.linalg.norm(canon_vec))
            sim = float(np.dot(vec, canon_vec) / norm_product) if norm_product > 0 else 0.0

            if sim > highest_sim:
                highest_sim = sim
                best_match = canon_name

        # 4. Collapse if cosine similarity exceeds tuned threshold
        if best_match and highest_sim >= self.threshold:
            self.canonical_map[raw_trimmed] = best_match
            if norm_stem:
                self.normalized_map[norm_stem] = best_match
            if raw_trimmed not in self.aliases[best_match]:
                self.aliases[best_match].append(raw_trimmed)
            return best_match

        # 5. Register new canonical node
        self.canonical_map[raw_trimmed] = raw_trimmed
        if norm_stem:
            self.normalized_map[norm_stem] = raw_trimmed
        self.canonical_embeddings[raw_trimmed] = vec
        self.aliases[raw_trimmed] = [raw_trimmed]
        return raw_trimmed

    def get_aliases(self, canonical_name: str) -> List[str]:
        return self.aliases.get(canonical_name, [canonical_name])
