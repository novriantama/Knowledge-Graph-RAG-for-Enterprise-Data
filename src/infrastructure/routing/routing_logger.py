import os
import json
import datetime
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RoutingLogger:
    """Persistent JSON Lines logger recording every query routing decision, execution outcome, and metrics for Phase 5 Benchmarking."""

    def __init__(self, log_file_path: str = "./cache/routing_decisions.jsonl"):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def log_decision(
        self,
        question: str,
        route: str,
        confidence: float,
        is_fallback: bool,
        target_entities: List[str],
        reasoning: str,
        graph_paths_count: int,
        vector_passages_count: int,
        retrieved_chunk_ids: List[str],
        citations: List[str],
        latency_ms: float
    ) -> Dict[str, Any]:
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "question": question,
            "route": route,
            "confidence": confidence,
            "is_fallback": is_fallback,
            "target_entities": target_entities,
            "reasoning": reasoning,
            "graph_paths_count": graph_paths_count,
            "vector_passages_count": vector_passages_count,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "citations": citations,
            "latency_ms": round(latency_ms, 2)
        }

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Routing decision logged for query '{question[:30]}...' -> Route: {route}")
        except Exception as e:
            logger.error(f"Failed to append routing log entry: {e}")

        return log_entry

    def load_logged_decisions(self) -> List[Dict[str, Any]]:
        """Loads all logged routing decisions for Phase 5 analysis."""
        if not os.path.exists(self.log_file_path):
            return []

        entries = []
        with open(self.log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    entries.append(json.loads(line_str))
        return entries
