import time
import json
import logging
from typing import List, Dict, Any, Tuple
from tabulate import tabulate
from anthropic import RateLimitError
from src.application.query_pipeline import QueryPipelineUseCase
from src.domain.interfaces import IVectorRepository, IGeneratorService
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class BenchmarkRunnerUseCase:
    """Stratified benchmark runner evaluating Plain Vector RAG vs. Hybrid KG-RAG across 1-hop, 2-hop, 3-hop, aggregation, and out-of-scope queries."""

    # Approximate API pricing models (Input $3/1M tokens, Output $15/1M tokens for Sonnet; Haiku Router $0.25/$1.25)
    PLAIN_VECTOR_COST_PER_QUERY = 0.0035 # ~$0.0035 per query
    KG_RAG_COST_PER_QUERY = 0.0058       # ~$0.0058 per query (Router + Graph traversal + Sonnet)
    ONE_TIME_GRAPH_INGESTION_COST = 0.07 # Estimated corpus extraction cost

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
        for attempt in range(1, 4):
            try:
                query_embedding = self.encoder.encode(query).tolist()
                passages = self.vector_repo.similarity_search(query_embedding, top_k=5)
                return self.generator.generate_grounded_answer(
                    query=query,
                    graph_paths=[],
                    vector_passages=passages,
                    route_choice="VECTOR"
                )
            except RateLimitError as rle:
                wait_sec = attempt * 10
                logger.warning(f"Benchmark rate limit (Vector RAG) attempt {attempt}/3. Waiting {wait_sec}s...")
                time.sleep(wait_sec)
        raise RuntimeError("Vector RAG rate limit retries exhausted.")

    def _run_kg_rag(self, query: str):
        for attempt in range(1, 4):
            try:
                return self.kg_pipeline.execute(query)
            except RateLimitError as rle:
                wait_sec = attempt * 10
                logger.warning(f"Benchmark rate limit (KG-RAG) attempt {attempt}/3. Waiting {wait_sec}s...")
                time.sleep(wait_sec)
        raise RuntimeError("KG-RAG rate limit retries exhausted.")

    def _evaluate_answer(self, generated_answer: str, expected_keywords: List[str], category: str) -> bool:
        if not expected_keywords:
            return True
        gen_lower = generated_answer.lower()
        if category == "out_of_scope":
            return any(kw.lower() in gen_lower for kw in expected_keywords)
        matches = sum(1 for kw in expected_keywords if kw.lower() in gen_lower)
        return (matches / len(expected_keywords)) >= 0.5

    def execute_benchmark(self, questions_json_path: str = "data/benchmark_questions.json") -> Dict[str, Any]:
        with open(questions_json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        results = []
        total_q = len(questions)

        for idx, q in enumerate(questions, 1):
            q_id = q.get("id")
            q_text = q["question"]
            category = q.get("category", "single_hop")
            hop_count = q.get("hop_count", 1)
            expected_keywords = q.get("expected_answer_keywords", [])

            logger.info(f"[{idx}/{total_q}] Benchmarking Q{q_id} ({category}): '{q_text}'")

            # 1. Plain Vector RAG
            t0 = time.time()
            vec_ans_obj = self._run_plain_vector_rag(q_text)
            vec_latency = time.time() - t0
            vec_correct = self._evaluate_answer(vec_ans_obj.answer, expected_keywords, category)
            time.sleep(1.5)

            # 2. Hybrid KG-RAG
            t0 = time.time()
            kg_ans_obj = self._run_kg_rag(q_text)
            kg_latency = time.time() - t0
            kg_correct = self._evaluate_answer(kg_ans_obj.answer, expected_keywords, category)
            time.sleep(1.5)

            results.append({
                "id": q_id,
                "category": category,
                "hop_count": hop_count,
                "vec_correct": vec_correct,
                "vec_latency": vec_latency,
                "kg_correct": kg_correct,
                "kg_latency": kg_latency
            })

        summary_table, delta_report = self._build_summary_table(results)
        return {
            "raw_results": results,
            "summary_table": summary_table,
            "delta_report": delta_report,
            "ingestion_cost_usd": self.ONE_TIME_GRAPH_INGESTION_COST
        }

    def _build_summary_table(self, results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        categories = ["single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope"]
        category_labels = {
          "single_hop": "1-Hop (Single Fact)",
          "two_hop": "2-Hop (Relational)",
          "three_hop": "3-Hop (Transitive Chain)",
          "aggregation": "Aggregation / Grouping",
          "out_of_scope": "Out of Scope (Refusal)"
        }

        summary_rows = []
        delta_report = {}

        for cat in categories:
            cat_qs = [r for r in results if r["category"] == cat]
            if not cat_qs:
                continue

            vec_acc = (sum(1 for r in cat_qs if r["vec_correct"]) / len(cat_qs)) * 100
            kg_acc = (sum(1 for r in cat_qs if r["kg_correct"]) / len(cat_qs)) * 100
            delta_acc = kg_acc - vec_acc

            vec_avg_lat = sum(r["vec_latency"] for r in cat_qs) / len(cat_qs)
            kg_avg_lat = sum(r["kg_latency"] for r in cat_qs) / len(cat_qs)

            delta_report[cat] = {
                "vector_acc_percent": round(vec_acc, 1),
                "kg_acc_percent": round(kg_acc, 1),
                "delta_acc_percent": round(delta_acc, 1),
                "vector_avg_latency_sec": round(vec_avg_lat, 2),
                "kg_avg_latency_sec": round(kg_avg_lat, 2)
            }

            summary_rows.append([
                category_labels.get(cat, cat),
                f"{vec_acc:.1f}%",
                f"{kg_acc:.1f}%",
                f"{'+' if delta_acc >= 0 else ''}{delta_acc:.1f}%",
                f"{vec_avg_lat:.2f}s",
                f"{kg_avg_lat:.2f}s",
                f"${self.PLAIN_VECTOR_COST_PER_QUERY:.4f}",
                f"${self.KG_RAG_COST_PER_QUERY:.4f}"
            ])

        headers = [
            "Query Complexity",
            "Vector RAG Acc",
            "Hybrid KG-RAG Acc",
            "Accuracy Delta",
            "Vector Latency",
            "KG-RAG Latency",
            "Vector Cost",
            "KG Cost"
        ]

        table_md = tabulate(summary_rows, headers=headers, tablefmt="github")
        return table_md, delta_report
