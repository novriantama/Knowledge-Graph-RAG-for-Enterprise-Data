import time
import json
from typing import List, Dict, Any
from tabulate import tabulate
from src.application.query_pipeline import QueryPipelineUseCase
from src.domain.interfaces import IVectorRepository, IGeneratorService
from sentence_transformers import SentenceTransformer

class BenchmarkRunnerUseCase:
    """Benchmark runner evaluating accuracy and latency across 1-hop, 2-hop, and 3-hop queries."""

    def __init__(
        self,
        kg_pipeline: QueryPipelineUseCase,
        vector_repo: IVectorRepository,
        generator: IGeneratorService
    ):
        self.kg_pipeline = kg_pipeline
        self.vector_repo = vector_repo
        self.generator = generator
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def _run_plain_vector_rag(self, query: str):
        query_embedding = self.encoder.encode(query).tolist()
        passages = self.vector_repo.similarity_search(query_embedding, top_k=5)
        return self.generator.generate_grounded_answer(
            query=query,
            graph_paths=[],
            vector_passages=passages,
            route_choice="VECTOR"
        )

    def execute_benchmark(self, questions_json_path: str) -> Dict[str, Any]:
        with open(questions_json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        results = []

        for q in questions:
            q_text = q["question"]
            hop_count = q.get("hop_count", 1)
            expected_keywords = q.get("expected_answer_keywords", [])

            # 1. Plain Vector RAG
            t0 = time.time()
            vec_answer = self._run_plain_vector_rag(q_text)
            vec_latency = time.time() - t0
            vec_correct = any(kw.lower() in vec_answer.answer.lower() for kw in expected_keywords)

            # 2. Hybrid KG-RAG
            t0 = time.time()
            kg_answer = self.kg_pipeline.execute(q_text)
            kg_latency = time.time() - t0
            kg_correct = any(kw.lower() in kg_answer.answer.lower() for kw in expected_keywords)

            results.append({
                "id": q.get("id"),
                "hop_count": hop_count,
                "vec_correct": vec_correct,
                "vec_latency": vec_latency,
                "kg_correct": kg_correct,
                "kg_latency": kg_latency
            })

        summary_table = self._build_summary_table(results)
        return {"raw_results": results, "summary_table": summary_table}

    def _build_summary_table(self, results: List[Dict[str, Any]]) -> str:
        hops = [1, 2, 3]
        summary_rows = []

        for h in hops:
            hop_qs = [r for r in results if r["hop_count"] == h]
            if not hop_qs:
                continue

            vec_acc = (sum(1 for r in hop_qs if r["vec_correct"]) / len(hop_qs)) * 100
            kg_acc = (sum(1 for r in hop_qs if r["kg_correct"]) / len(hop_qs)) * 100
            vec_avg_lat = sum(r["vec_latency"] for r in hop_qs) / len(hop_qs)
            kg_avg_lat = sum(r["kg_latency"] for r in hop_qs) / len(hop_qs)

            summary_rows.append([
                f"{h}-Hop",
                f"{vec_acc:.1f}%",
                f"{kg_acc:.1f}%",
                f"+{kg_acc - vec_acc:.1f}%",
                f"{vec_avg_lat:.2f}s",
                f"{kg_avg_lat:.2f}s"
            ])

        headers = ["Complexity", "Vector RAG Acc", "KG-RAG Acc", "Delta", "Vector Latency", "KG-RAG Latency"]
        return tabulate(summary_rows, headers=headers, tablefmt="github")
