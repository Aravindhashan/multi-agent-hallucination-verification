"""Diagnostic Experiment: Separating NLI Errors from Retrieval Errors on SciFact Benchmark.

Takes 100 real claims with gold evidence from SciFact.
1. Evaluates NLI alone on gold evidence passages (isolates pure model reasoning capability).
2. Evaluates the full pipeline on retrieved evidence passages (tests retrieval + model).
3. Computes the diagnostic performance gap and partitions errors into:
   - Pure Retrieval Failures (Gold Correct, Retrieved Incorrect)
   - Pure Model Failures (Gold Incorrect, Retrieved Incorrect)
   - End-to-End Successes (Both Correct)
   - Fortuitous / Ambiguous (Gold Incorrect, Retrieved Correct)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure backend root is in sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import os
import torch

# Set PyTorch CPU threads for maximum throughput on multi-threaded CPU
num_threads = min(8, os.cpu_count() or 8)
torch.set_num_threads(num_threads)

from app.services.bm25_retriever import BM25Retriever
from app.services.corpus_loader import CorpusLoader
from app.services.corpus_schema import CorpusDocument
from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine
from app.services.nli_models import DocumentNLIPoolResult, NLILabel, SentencePrediction
from app.services.sentence_window import select_best_sentence_window, split_into_sentences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nli_vs_retrieval_gap")


def load_dataset_and_corpus(
    claims_path: Path,
    corpus_path: Path,
    sample_size: int = 100,
    balance_classes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, CorpusDocument], BM25Retriever]:
    """Load SciFact corpus, index into BM25, and extract claims with gold evidence."""
    logger.info("Loading SciFact corpus from %s...", corpus_path)
    loader = CorpusLoader.from_file(corpus_path)
    corpus_map: dict[str, CorpusDocument] = {doc.id: doc for doc in loader.documents}
    logger.info("Indexed %d corpus documents. Building BM25 index...", len(corpus_map))
    bm25 = BM25Retriever.from_documents(loader.documents)

    logger.info("Loading SciFact claims from %s...", claims_path)
    claims_with_evidence: list[dict[str, Any]] = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Must have cited gold evidence doc IDs and raw evidence
            if data.get("evidence_doc_ids") and data.get("raw_evidence"):
                # Determine primary label
                raw_ev = data["raw_evidence"]
                primary_label = None
                for doc_key, ev_list in raw_ev.items():
                    for ev in ev_list:
                        lbl = ev.get("label", "").upper()
                        if lbl in ("SUPPORT", "CONTRADICT"):
                            primary_label = lbl
                            break
                    if primary_label:
                        break
                if primary_label:
                    data["gold_label"] = primary_label
                    claims_with_evidence.append(data)

    logger.info("Found %d total claims with gold evidence.", len(claims_with_evidence))

    selected_claims: list[dict[str, Any]] = []
    if balance_classes:
        supports = [c for c in claims_with_evidence if c["gold_label"] == "SUPPORT"]
        contradicts = [c for c in claims_with_evidence if c["gold_label"] == "CONTRADICT"]
        half = sample_size // 2
        selected_claims = supports[:half] + contradicts[:half]
        logger.info(
            "Selected balanced sample of %d claims: %d SUPPORT, %d CONTRADICT",
            len(selected_claims),
            len(supports[:half]),
            len(contradicts[:half]),
        )
    else:
        selected_claims = claims_with_evidence[:sample_size]
        logger.info("Selected first %d claims with gold evidence.", len(selected_claims))

    return selected_claims, corpus_map, bm25


def extract_gold_evidence_text(claim: dict[str, Any], corpus_map: dict[str, CorpusDocument]) -> str:
    """Extract gold sentence text from corpus documents using annotated sentence indices."""
    gold_sentences: list[str] = []
    raw_ev = claim.get("raw_evidence", {})

    for doc_id_clean, ev_list in raw_ev.items():
        doc_key = f"scifact_{doc_id_clean}" if not doc_id_clean.startswith("scifact_") else doc_id_clean
        doc = corpus_map.get(doc_key)
        if not doc:
            continue
        all_sents = split_into_sentences(doc.text)
        for ev in ev_list:
            s_indices = ev.get("sentences", [])
            for idx in s_indices:
                if 0 <= idx < len(all_sents):
                    sent_text = all_sents[idx].strip()
                    if sent_text and sent_text not in gold_sentences:
                        gold_sentences.append(sent_text)

    if not gold_sentences:
        # Fallback to full document text of first cited doc
        first_doc_id = claim["evidence_doc_ids"][0]
        if first_doc_id in corpus_map:
            return corpus_map[first_doc_id].text
        return ""

    return " ".join(gold_sentences)


def run_experiment(
    claims: list[dict[str, Any]],
    corpus_map: dict[str, CorpusDocument],
    bm25: BM25Retriever,
    top_k: int = 3,
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    batch_size: int = 16,
    sentence_level: bool = True,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Execute both Gold NLI and Retrieved NLI, computing diagnostic breakdown.

    When sentence_level=True (default SciFact standard):
    - Scores every sentence of the abstract against the claim.
    - Max-pools entailment and contradiction probabilities across sentences.
    - Removes heuristic window misses completely and evaluates top-K retrieval depth.
    """
    nli_settings = NLISettings(
        model_name=model_name,
        device="cpu",
        batch_size=batch_size,
        enable_sentence_windows=not sentence_level,
        sentence_window_strategy="hybrid",
    )
    engine = NLIEngine(nli_settings)

    detailed_results: list[dict[str, Any]] = []
    gold_correct_count = 0
    retrieved_correct_count = 0
    retrieved_at_1_correct_count = 0
    recall_hit_count = 0
    recall_at_1_count = 0

    gold_confusions: dict[str, dict[str, int]] = {
        "SUPPORT": {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0},
        "CONTRADICT": {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0},
    }
    ret_confusions: dict[str, dict[str, int]] = {
        "SUPPORT": {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0},
        "CONTRADICT": {"ENTAILMENT": 0, "NEUTRAL": 0, "CONTRADICTION": 0},
    }

    # =========================================================================
    # STEP 1: Gold Evidence Evaluation (Tests Pure Model Reasoning)
    # =========================================================================
    logger.info("=== STEP 1: Running Model-Only Evaluation on Gold Evidence (%s) ===", model_name)
    logger.info("Evaluation Mode: %s", "Sentence-Level NLI with Max-Pooling" if sentence_level else "Legacy Heuristic Windowing")
    t0_gold = time.perf_counter()

    gold_pool_results: list[DocumentNLIPoolResult | NLIResult] = []

    if sentence_level:
        for idx, claim in enumerate(claims):
            claim_text = claim["claim_text"]
            cited_ids = claim["evidence_doc_ids"]
            gold_docs = [(d_id, corpus_map[d_id].text) for d_id in cited_ids if d_id in corpus_map]
            pool_res = engine.predict_multi_document_sentences(claim_text, gold_docs, threshold=threshold)
            gold_pool_results.append(pool_res)

            if (idx + 1) % 25 == 0 or idx == len(claims) - 1:
                logger.info("Gold NLI progress: %d/%d claims evaluated (%.2fs)", idx + 1, len(claims), time.perf_counter() - t0_gold)
    else:
        gold_pairs: list[tuple[str, str]] = []
        for claim in claims:
            gold_premise = extract_gold_evidence_text(claim, corpus_map)
            gold_pairs.append((claim["claim_text"], gold_premise))
        gold_pool_results = engine.predict_batch(gold_pairs)

    t1_gold = time.perf_counter()
    logger.info("Evaluated %d gold claims in %.2fs", len(claims), t1_gold - t0_gold)

    # Interim Gold accuracy computation
    for idx, claim in enumerate(claims):
        gold_target_nli = "ENTAILMENT" if claim["gold_label"] == "SUPPORT" else "CONTRADICTION"
        if sentence_level:
            gold_pred = gold_pool_results[idx].winning_label.value.upper()
        else:
            gold_pred = gold_pool_results[idx].label.value.upper()
        if gold_pred == gold_target_nli:
            gold_correct_count += 1
        gold_confusions[claim["gold_label"]][gold_pred] = gold_confusions[claim["gold_label"]].get(gold_pred, 0) + 1

    logger.info(">>> GOLD NLI ACCURACY: %d/%d (%.2f%%) <<<", gold_correct_count, len(claims), 100.0 * gold_correct_count / len(claims))

    # =========================================================================
    # STEP 2: Retrieved Evidence Evaluation (top_k documents with sentence max-pooling)
    # =========================================================================
    logger.info("=== STEP 2: Running Full Pipeline on Retrieved Evidence (top_k=%d, Sentence-Level Max-Pooling) ===", top_k)
    retrieval_records: list[dict[str, Any]] = []
    ret_pool_results: list[DocumentNLIPoolResult | str] = []
    ret_pool_at_1_results: list[DocumentNLIPoolResult | str] = []

    t0_retrieval = time.perf_counter()

    for c_idx, claim in enumerate(claims):
        claim_text = claim["claim_text"]
        gold_ids = set(claim["evidence_doc_ids"])

        # BM25 Lexical retrieval over 5,183 SciFact abstracts
        results = bm25.retrieve(claim_text, top_k=top_k)
        retrieved_ids = [r.document_id for r in results]

        recall_hit = any(g_id in retrieved_ids for g_id in gold_ids)
        rank_hit = None
        for r_idx, r_id in enumerate(retrieved_ids):
            if r_id in gold_ids:
                rank_hit = r_idx + 1
                break

        if recall_hit:
            recall_hit_count += 1
        if rank_hit == 1:
            recall_at_1_count += 1

        claim_record = {
            "claim_id": claim["claim_id"],
            "retrieved_doc_ids": retrieved_ids,
            "gold_doc_ids": list(gold_ids),
            "recall_hit": recall_hit,
            "gold_rank": rank_hit,
            "retrieved_results": results,
        }
        retrieval_records.append(claim_record)

        if sentence_level:
            cand_docs = [(r.document_id, r.text) for r in results]
            # Max-pool across all top-K documents
            pool_topk = engine.predict_multi_document_sentences(claim_text, cand_docs, threshold=threshold)
            ret_pool_results.append(pool_topk)

            # Also max-pool over top-1 document only for explicit Top-1 vs Top-K comparison
            if cand_docs:
                pool_top1 = engine.predict_multi_document_sentences(claim_text, cand_docs[:1], threshold=threshold)
                ret_pool_at_1_results.append(pool_top1)
            else:
                ret_pool_at_1_results.append(pool_topk)
        else:
            cand_pairs = [(claim_text, r.text) for r in results]
            c_res = engine.predict_batch(cand_pairs)
            best_cand = None
            best_decisive = -1.0
            for item in c_res:
                dec = max(item.p_entailment, item.p_contradiction)
                if dec > best_decisive:
                    best_decisive = dec
                    best_cand = item
            if best_cand and best_cand.label in (NLILabel.ENTAILMENT, NLILabel.CONTRADICTION):
                r_pred = best_cand.label.value.upper()
            else:
                r_pred = "NEUTRAL"
            ret_pool_results.append(r_pred)

        if (c_idx + 1) % 25 == 0 or c_idx == len(claims) - 1:
            logger.info("Retrieval + NLI progress: %d/%d claims evaluated (%.2fs)", c_idx + 1, len(claims), time.perf_counter() - t0_retrieval)

    t1_retrieval = time.perf_counter()
    logger.info("Completed retrieval pipeline in %.2fs", t1_retrieval - t0_retrieval)

    # =========================================================================
    # STEP 3: Error Partitioning & Performance Gap
    # =========================================================================
    logger.info("=== STEP 3: Computing Diagnostic Metrics and Partitioning Errors ===")
    both_correct = 0
    retrieval_failure = 0  # Gold correct, Retrieved incorrect
    model_failure = 0      # Gold incorrect, Retrieved incorrect
    fortuitous = 0         # Gold incorrect, Retrieved correct

    for i, claim in enumerate(claims):
        gold_label = claim["gold_label"]
        gold_target_nli = "ENTAILMENT" if gold_label == "SUPPORT" else "CONTRADICTION"

        if sentence_level:
            g_pool = gold_pool_results[i]
            gold_pred = g_pool.winning_label.value.upper()
            gold_probs = {
                "entailment": round(g_pool.p_entailment_max, 4),
                "contradiction": round(g_pool.p_contradiction_max, 4),
            }
            gold_winning_sent = g_pool.winning_sentence.sentence_text if g_pool.winning_sentence else None

            r_pool = ret_pool_results[i]
            ret_pred = r_pool.winning_label.value.upper()
            ret_probs = {
                "entailment": round(r_pool.p_entailment_max, 4),
                "contradiction": round(r_pool.p_contradiction_max, 4),
            }
            ret_winning_sent = r_pool.winning_sentence.sentence_text if r_pool.winning_sentence else None
            ret_winning_doc = r_pool.winning_sentence.doc_id if r_pool.winning_sentence else None

            r1_pool = ret_pool_at_1_results[i]
            ret_pred_at_1 = r1_pool.winning_label.value.upper()
        else:
            gold_res = gold_pool_results[i]
            gold_pred = gold_res.label.value.upper()
            gold_probs = {
                "entailment": round(gold_res.p_entailment, 4),
                "neutral": round(gold_res.p_neutral, 4),
                "contradiction": round(gold_res.p_contradiction, 4),
            }
            gold_winning_sent = None
            ret_pred = ret_pool_results[i]
            ret_pred_at_1 = ret_pred
            ret_probs = {}
            ret_winning_sent = None
            ret_winning_doc = None

        gold_is_correct = (gold_pred == gold_target_nli)
        ret_is_correct = (ret_pred == gold_target_nli)
        ret_at_1_is_correct = (ret_pred_at_1 == gold_target_nli)

        if ret_is_correct:
            retrieved_correct_count += 1
        if ret_at_1_is_correct:
            retrieved_at_1_correct_count += 1

        ret_confusions[gold_label][ret_pred] = ret_confusions[gold_label].get(ret_pred, 0) + 1

        if gold_is_correct and ret_is_correct:
            category = "BOTH_CORRECT"
            both_correct += 1
        elif gold_is_correct and not ret_is_correct:
            category = "RETRIEVAL_FAILURE"
            retrieval_failure += 1
        elif not gold_is_correct and not ret_is_correct:
            category = "MODEL_FAILURE"
            model_failure += 1
        else:
            category = "FORTUITOUS"
            fortuitous += 1

        detailed_results.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "gold_label": gold_label,
            "gold_doc_ids": claim["evidence_doc_ids"],
            "gold_prediction": gold_pred,
            "gold_is_correct": gold_is_correct,
            "gold_probs": gold_probs,
            "gold_winning_sentence": gold_winning_sent,
            "retrieval_recall_hit": retrieval_records[i]["recall_hit"],
            "retrieval_gold_rank": retrieval_records[i]["gold_rank"],
            "retrieved_doc_ids": retrieval_records[i]["retrieved_doc_ids"],
            "retrieved_prediction_top_k": ret_pred,
            "retrieved_is_correct_top_k": ret_is_correct,
            "retrieved_prediction_at_1": ret_pred_at_1,
            "retrieved_is_correct_at_1": ret_at_1_is_correct,
            "retrieved_probs": ret_probs,
            "retrieved_winning_sentence": ret_winning_sent,
            "retrieved_winning_doc_id": ret_winning_doc,
            "category": category,
        })

    n = len(claims)
    gold_acc = gold_correct_count / n
    ret_acc = retrieved_correct_count / n
    ret_acc_1 = retrieved_at_1_correct_count / n
    gap = gold_acc - ret_acc
    recall_at_k = recall_hit_count / n
    recall_at_1 = recall_at_1_count / n

    summary = {
        "model_name": model_name,
        "evaluation_strategy": "sentence_level_max_pooling" if sentence_level else "heuristic_windowing",
        "retrieval_top_k": top_k,
        "total_claims": n,
        "gold_accuracy": round(gold_acc, 4),
        "retrieved_accuracy_top_k": round(ret_acc, 4),
        "retrieved_accuracy_top_1": round(ret_acc_1, 4),
        "retrieval_depth_delta": round((ret_acc - ret_acc_1) * 100, 2),
        "accuracy_gap": round(gap, 4),
        "retrieval_recall_at_top_k": round(recall_at_k, 4),
        "retrieval_recall_at_1": round(recall_at_1, 4),
        "error_partition": {
            "both_correct_count": both_correct,
            "both_correct_pct": round(both_correct / n * 100, 2),
            "retrieval_failure_count": retrieval_failure,
            "retrieval_failure_pct": round(retrieval_failure / n * 100, 2),
            "model_failure_count": model_failure,
            "model_failure_pct": round(model_failure / n * 100, 2),
            "fortuitous_count": fortuitous,
            "fortuitous_pct": round(fortuitous / n * 100, 2),
        },
        "gold_confusion_matrix": gold_confusions,
        "retrieved_confusion_matrix": ret_confusions,
        "timing": {
            "gold_inference_sec": round(t1_gold - t0_gold, 2),
            "retrieval_pipeline_sec": round(t1_retrieval - t0_retrieval, 2),
        },
    }

    return {
        "summary": summary,
        "detailed_results": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sentence-Level NLI with Max-Pooling on SciFact")
    parser.add_argument("--claims-path", type=str, default="data/scifact_claims.jsonl")
    parser.add_argument("--corpus-path", type=str, default="data/scifact_corpus.jsonl")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved documents to evaluate (e.g. 3 or 5)")
    parser.add_argument("--unbalanced", action="store_true")
    parser.add_argument("--model-name", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heuristic-window", action="store_true", help="Use legacy heuristic windowing instead of sentence-level NLI")
    parser.add_argument("--output-json", type=str, default="data/nli_vs_retrieval_gap_report_sentence_nli.json")
    parser.add_argument("--baseline-report", type=str, default="data/nli_vs_retrieval_gap_report.json")
    args = parser.parse_args()

    claims, corpus_map, bm25 = load_dataset_and_corpus(
        claims_path=Path(args.claims_path),
        corpus_path=Path(args.corpus_path),
        sample_size=args.sample_size,
        balance_classes=not args.unbalanced,
    )

    sentence_level = not args.heuristic_window

    report = run_experiment(
        claims=claims,
        corpus_map=corpus_map,
        bm25=bm25,
        top_k=args.top_k,
        model_name=args.model_name,
        batch_size=args.batch_size,
        sentence_level=sentence_level,
    )

    summary = report["summary"]
    print("\n" + "=" * 80)
    print("   DIAGNOSTIC GAP ANALYSIS: SENTENCE-LEVEL NLI MAX-POOLING ON SCIFACT   ")
    print("=" * 80)
    print(f"Evaluated Model               : {summary['model_name']}")
    print(f"Evaluation Strategy           : {summary['evaluation_strategy']}")
    print(f"Sample Size                   : {summary['total_claims']} real SciFact claims")
    print(f"Gold Evidence Accuracy        : {summary['gold_accuracy'] * 100:.2f}%  (Tests ONLY Model with Max-Pool)")
    print(f"Retrieved Accuracy @ Top-{args.top_k}  : {summary['retrieved_accuracy_top_k'] * 100:.2f}%  (Top-{args.top_k} Docs Sentence Max-Pool)")
    print(f"Retrieved Accuracy @ Top-1    : {summary['retrieved_accuracy_top_1'] * 100:.2f}%  (Top-1 Doc Only)")
    print(f"Retrieval Depth Gain          : {summary['retrieval_depth_delta']:+.2f}%  (Benefit of Top-{args.top_k} vs Top-1)")
    print(f"The Diagnostic Gap (Delta)    : {summary['accuracy_gap'] * 100:+.2f}%")
    print(f"BM25 Retrieval Recall@{args.top_k}      : {summary['retrieval_recall_at_top_k'] * 100:.2f}%")
    print(f"BM25 Retrieval Recall@1       : {summary['retrieval_recall_at_1'] * 100:.2f}%")
    print("-" * 80)
    print("ERROR PARTITIONING:")
    ep = summary["error_partition"]
    print(f"  [1] Both Correct (End-to-End Success)     : {ep['both_correct_count']:3d} ({ep['both_correct_pct']}%)")
    print(f"  [2] Retrieval Failures (Gold OK, Ret Bad) : {ep['retrieval_failure_count']:3d} ({ep['retrieval_failure_pct']}%)  <-- Retrieval Starvation")
    print(f"  [3] Model Failures (Gold Bad, Ret Bad)     : {ep['model_failure_count']:3d} ({ep['model_failure_pct']}%)  <-- NLI Reasoning Gap")
    print(f"  [4] Fortuitous / Outlier                  : {ep['fortuitous_count']:3d} ({ep['fortuitous_pct']}%)")
    print("-" * 80)

    # Check for baseline report comparison
    base_path = Path(args.baseline_report)
    if base_path.exists():
        try:
            with open(base_path, "r", encoding="utf-8") as bf:
                base_data = json.load(bf)
                base_s = base_data.get("summary", {})
                if base_s:
                    b_gold = base_s.get("gold_accuracy", 0.0) * 100
                    b_ret = base_s.get("retrieved_accuracy", 0.0) * 100
                    b_ep = base_s.get("error_partition", {})
                    b_mod = b_ep.get("model_failure_pct", 0.0)
                    b_both = b_ep.get("both_correct_pct", 0.0)

                    c_gold = summary["gold_accuracy"] * 100
                    c_ret = summary["retrieved_accuracy_top_k"] * 100
                    c_mod = ep["model_failure_pct"]
                    c_both = ep["both_correct_pct"]

                    print("STRATEGY COMPARISON (BASELINE HEURISTIC WINDOW VS SENTENCE-LEVEL NLI):")
                    print(f"  Metric                         | Baseline (Heuristic Window) | Sentence-Level (Top-{args.top_k}) | Delta")
                    print(f"  -------------------------------+-----------------------------+-----------------------+-------")
                    print(f"  Gold Accuracy (Model Only)     | {b_gold:6.2f}%                      | {c_gold:6.2f}%                 | {c_gold - b_gold:+6.2f}%")
                    print(f"  Retrieved Accuracy (Full)      | {b_ret:6.2f}%                      | {c_ret:6.2f}%                 | {c_ret - b_ret:+6.2f}%")
                    print(f"  End-to-End Success Rate        | {b_both:6.2f}%                      | {c_both:6.2f}%                 | {c_both - b_both:+6.2f}%")
                    print(f"  Pure Model Failures            | {b_mod:6.2f}%                      | {c_mod:6.2f}%                 | {c_mod - b_mod:+6.2f}%")
                    print("-" * 80)
        except Exception as e:
            logger.warning("Could not read baseline report %s: %s", base_path, e)

    print("CONFUSION MATRIX ON GOLD EVIDENCE (NLI ALONE):")
    for true_lbl, preds in summary["gold_confusion_matrix"].items():
        print(f"  True {true_lbl:10s} -> Entailment: {preds.get('ENTAILMENT', 0):2d}, Neutral: {preds.get('NEUTRAL', 0):2d}, Contradiction: {preds.get('CONTRADICTION', 0):2d}")
    print("-" * 80)
    print(f"CONFUSION MATRIX ON RETRIEVED EVIDENCE (TOP-{args.top_k} + MODEL):")
    for true_lbl, preds in summary["retrieved_confusion_matrix"].items():
        print(f"  True {true_lbl:10s} -> Entailment: {preds.get('ENTAILMENT', 0):2d}, Neutral: {preds.get('NEUTRAL', 0):2d}, Contradiction: {preds.get('CONTRADICTION', 0):2d}")
    print("=" * 80 + "\n")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved full diagnostic report to %s", out_path)


if __name__ == "__main__":
    main()

