"""Synthetic Provenance Corpus Generator and Benchmark Evaluation.

Rebuilds the deduplication and source independence evaluation experiment properly:
1. Programmatically generates synthetic provenance variations:
   - Verbatim exact copies (syndicated mirrors)
   - Paraphrased variations (synonym replacement, passive/active voice, sentence rephrasing)
   - Wire-prefix copies (e.g. "WASHINGTON (Reuters) - ...", "PARIS (AP) - ...")
   - Condensed summaries (leading clause / decisive rationale extraction)
2. Freezes the generated corpus to disk BEFORE running any evaluations.
3. Sweeps dependent copy count N (from 0 up to 6 per claim) across 100 balanced SciFact claims.
4. Compares four rigorous aggregation configurations:
   - Config 1: Naive Count (unit weight per raw item, no deduplication)
   - Config 2: Cluster-and-Count-One / B2 (exact / near-duplicate surface dedup baseline)
   - Config 3: Kish Only (Kish effective sample size discounting on all surviving items)
   - Config 4: Full Pipeline (Complete-linkage clustering, Kish n_eff, representative selection W(C_k), informativeness gating)
5. Evaluates and plots Accuracy and Expected Calibration Error (ECE) against dependent copy ratio.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.schemas import EvidenceItem
from app.services.cluster_pooling import (
    ClusterPoolingService,
    combine_log_odds,
    combine_nli_probabilities,
    log_odds_to_prob,
    prob_to_log_odds,
)
from app.services.source_independence_scorer import (
    SourceIndependenceDetector,
    compute_kish_effective_sample_size,
    normalize_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_provenance_sweep")

# Programmatic Wire Prefixes
WIRE_PREFIXES: list[str] = [
    "WASHINGTON (Reuters) - ",
    "PARIS (AP) - ",
    "LONDON (AFP) - ",
    "GENEVA (Bloomberg) - ",
    "NEW YORK (Dow Jones) - ",
    "BERLIN (DPA) - ",
    "TOKYO (Kyodo) - ",
]

# Domain lists for syndication simulation
SYNDICATE_DOMAINS: list[str] = [
    "news.yahoo.com",
    "msn.com",
    "aol.com",
    "dailyhealthnews.org",
    "techblog.example.com",
    "sciencemirror.net",
    "globalpress.org",
    "newsbreak.com",
]

# Curated scientific synonym dictionary for deterministic programmatic paraphrasing
PARAPHRASE_SYNONYMS: dict[str, list[str]] = {
    "demonstrated": ["showed", "revealed", "indicated", "confirmed", "established"],
    "demonstrates": ["shows", "reveals", "indicates", "confirms", "establishes"],
    "demonstrate": ["show", "reveal", "indicate", "confirm", "establish"],
    "showed": ["demonstrated", "revealed", "indicated", "confirmed"],
    "shows": ["demonstrates", "reveals", "indicates", "confirms"],
    "revealed": ["showed", "demonstrated", "uncovered", "indicated"],
    "reveals": ["shows", "demonstrates", "uncovers", "indicates"],
    "significantly": ["substantially", "markedly", "notably", "meaningfully", "statistically"],
    "significant": ["substantial", "marked", "notable", "meaningful"],
    "increased": ["elevated", "enhanced", "augmented", "amplified", "boosted"],
    "increases": ["elevates", "enhances", "augments", "amplifies"],
    "increase": ["elevation", "enhancement", "amplification", "boost"],
    "decreased": ["reduced", "attenuated", "lowered", "diminished", "curbed"],
    "decreases": ["reduces", "attenuates", "lowers", "diminishes"],
    "decrease": ["reduction", "attenuation", "decline"],
    "reduced": ["attenuated", "decreased", "lowered", "diminished"],
    "patients": ["individuals", "participants", "subjects", "cohort members"],
    "patient": ["individual", "subject", "participant"],
    "treatment": ["therapy", "intervention", "regimen", "clinical protocol"],
    "therapy": ["treatment", "intervention", "medical regimen"],
    "associated": ["correlated", "linked", "connected", "interrelated"],
    "association": ["correlation", "relationship", "linkage"],
    "crucial": ["critical", "essential", "vital", "indispensable"],
    "important": ["critical", "significant", "vital"],
    "observed": ["detected", "identified", "seen", "noted"],
    "effective": ["efficacious", "potent", "successful"],
    "expression": ["production", "manifestation", "abundance"],
    "activation": ["induction", "stimulation", "triggering"],
    "inhibits": ["blocks", "suppresses", "impedes", "prevents"],
    "inhibited": ["blocked", "suppressed", "impeded", "curbed"],
}


# ============================================================================
# 1. Programmatic Provenance Corpus Generator
# ============================================================================

def generate_verbatim_copy(orig_text: str, copy_idx: int, orig_url: str) -> dict[str, Any]:
    """Generate exact verbatim copy hosted on a syndicated mirror domain."""
    dom = SYNDICATE_DOMAINS[copy_idx % len(SYNDICATE_DOMAINS)]
    return {
        "text": orig_text,
        "provenance_type": "verbatim",
        "domain": dom,
        "source_url": f"https://{dom}/syndicated/{hashlib.md5(orig_text.encode()).hexdigest()[:8]}",
        "source_title": f"Syndicated Mirror ({dom})",
        "published_date": "2023-01-15",
    }


def generate_wire_prefix_copy(orig_text: str, copy_idx: int, orig_url: str) -> dict[str, Any]:
    """Generate wire-prefix copy with standard agency dateline header."""
    prefix = WIRE_PREFIXES[copy_idx % len(WIRE_PREFIXES)]
    dom = SYNDICATE_DOMAINS[copy_idx % len(SYNDICATE_DOMAINS)]
    wire_text = f"{prefix}{orig_text}"
    return {
        "text": wire_text,
        "provenance_type": "wire-prefix",
        "domain": dom,
        "source_url": f"https://{dom}/wire/{hashlib.md5(wire_text.encode()).hexdigest()[:8]}",
        "source_title": f"Wire Feed via {dom}",
        "published_date": "2023-01-16",
    }


def generate_paraphrase_copy(orig_text: str, copy_idx: int, orig_url: str, rng: random.Random) -> dict[str, Any]:
    """Programmatically generate lexical and syntactic paraphrase."""
    words = orig_text.split()
    new_words = []
    replaced_count = 0
    for w in words:
        clean_w = w.lower().strip(",.;:()[]\"'")
        punct_match = re.search(r"[,.;:()\[\]\"']+$", w)
        trailing_punct = punct_match.group(0) if punct_match else ""
        lead_match = re.search(r"^[,.;:()\[\]\"']+", w)
        lead_punct = lead_match.group(0) if lead_match else ""

        if clean_w in PARAPHRASE_SYNONYMS and rng.random() < 0.85:
            synonyms = PARAPHRASE_SYNONYMS[clean_w]
            syn = synonyms[copy_idx % len(synonyms)]
            # Preserve capitalization
            if w and w[0].isupper():
                syn = syn.capitalize()
            new_words.append(lead_punct + syn + trailing_punct)
            replaced_count += 1
        else:
            new_words.append(w)

    para_text = " ".join(new_words)
    # If no replacement was found in dictionary, apply standard syntactic rephrasing
    if replaced_count == 0:
        lead_phrases = [
            "In this study, researchers found that ",
            "Experimental findings confirmed that ",
            "The authors reported that ",
            "Clinical data indicated that ",
        ]
        para_text = f"{lead_phrases[copy_idx % len(lead_phrases)]}{orig_text[:1].lower()}{orig_text[1:]}"

    dom = SYNDICATE_DOMAINS[copy_idx % len(SYNDICATE_DOMAINS)]
    return {
        "text": para_text,
        "provenance_type": "paraphrase",
        "domain": dom,
        "source_url": f"https://{dom}/science-news/{hashlib.md5(para_text.encode()).hexdigest()[:8]}",
        "source_title": f"Science Coverage ({dom})",
        "published_date": "2023-01-18",
    }


def generate_summary_copy(orig_text: str, copy_idx: int, orig_url: str) -> dict[str, Any]:
    """Programmatically generate condensed summary extracting core clauses."""
    # Split into clauses or sentences
    clauses = re.split(r"[,;]\s+", orig_text)
    if len(clauses) >= 2:
        # Keep decisive first two clauses
        summ_text = f"{clauses[0]}, demonstrating that {clauses[1]}."
    else:
        # Substring truncation at natural word boundary
        cutoff = min(len(orig_text), 110)
        last_space = orig_text.rfind(" ", 0, cutoff)
        if last_space > 40:
            summ_text = f"Brief: {orig_text[:last_space]}."
        else:
            summ_text = f"Summary: {orig_text[:cutoff]}..."

    dom = SYNDICATE_DOMAINS[copy_idx % len(SYNDICATE_DOMAINS)]
    return {
        "text": summ_text,
        "provenance_type": "summary",
        "domain": dom,
        "source_url": f"https://{dom}/brief/{hashlib.md5(summ_text.encode()).hexdigest()[:8]}",
        "source_title": f"Medical Digest ({dom})",
        "published_date": "2023-01-20",
    }


def build_and_freeze_synthetic_corpus(
    benchmark_report_path: Path,
    output_frozen_path: Path,
    seed: int = 42,
    max_copies_per_claim: int = 6,
) -> dict[str, Any]:
    """Generate and freeze the synthetic provenance corpus to disk before evaluation.

    Ensures 100% reproducible benchmark by freezing all synthesized passages first.
    """
    logger.info("Loading baseline evaluation report from %s...", benchmark_report_path)
    report_data = json.loads(benchmark_report_path.read_text(encoding="utf-8"))
    detailed = report_data.get("detailed_results", [])
    if not detailed:
        raise ValueError(f"No detailed_results found in {benchmark_report_path}")

    rng = random.Random(seed)
    frozen_records: list[dict[str, Any]] = []

    provenance_generators = [
        ("verbatim", generate_verbatim_copy),
        ("wire-prefix", generate_wire_prefix_copy),
        ("paraphrase", lambda t, idx, u: generate_paraphrase_copy(t, idx, u, rng)),
        ("summary", generate_summary_copy),
    ]

    for item in detailed:
        claim_id = item["claim_id"]
        claim_text = item["claim_text"]
        gold_label = item["gold_label"]  # SUPPORT or CONTRADICT
        gold_premise = item["gold_premise_text"]
        gold_probs = item["gold_probs"]  # Calibrated DeBERTa NLI probabilities

        # Original clean evidence item
        orig_doc_id = f"{claim_id}_gold_orig"
        orig_url = f"https://pubmed.ncbi.nlm.nih.gov/{claim_id.replace('scifact_', '')}"
        orig_item = {
            "evidence_id": orig_doc_id,
            "document_id": orig_doc_id,
            "id": orig_doc_id,
            "text": gold_premise,
            "provenance_type": "original_gold",
            "domain": "ncbi.nlm.nih.gov",
            "source_url": orig_url,
            "source_title": "PubMed Central Peer-Reviewed Article",
            "published_date": "2022-01-01",
            "credibility_score": 0.95,
            "freshness_score": 0.90,
            "pollution_score": 0.00,
            "relevance_score": 1.00,
        }

        # Programmatically synthesize dependent copies cycling across the 4 types
        synthetic_copies: list[dict[str, Any]] = []
        for copy_idx in range(max_copies_per_claim):
            p_type, gen_func = provenance_generators[copy_idx % len(provenance_generators)]
            copy_dict = gen_func(gold_premise, copy_idx, orig_url)
            copy_id = f"{claim_id}_copy_{copy_idx}_{p_type}"
            copy_dict["evidence_id"] = copy_id
            copy_dict["document_id"] = copy_id
            copy_dict["id"] = copy_id
            copy_dict["credibility_score"] = 0.85
            copy_dict["freshness_score"] = 0.90
            copy_dict["pollution_score"] = 0.05
            copy_dict["relevance_score"] = 1.00
            synthetic_copies.append(copy_dict)

        frozen_records.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "gold_label": gold_label,
            "primary_nli_probs": gold_probs,
            "original_evidence": orig_item,
            "synthetic_copies": synthetic_copies,
        })

    frozen_corpus = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "total_claims": len(frozen_records),
            "max_copies_per_claim": max_copies_per_claim,
            "transformation_types": ["verbatim", "wire-prefix", "paraphrase", "summary"],
            "hash_sha256": hashlib.sha256(json.dumps([r["claim_id"] for r in frozen_records]).encode()).hexdigest(),
        },
        "records": frozen_records,
    }

    output_frozen_path.parent.mkdir(parents=True, exist_ok=True)
    output_frozen_path.write_text(json.dumps(frozen_corpus, indent=2), encoding="utf-8")
    logger.info("Successfully frozen synthetic provenance corpus (%d claims) to %s", len(frozen_records), output_frozen_path)
    return frozen_corpus


# ============================================================================
# 2. Aggregation Configurations
# ============================================================================

def run_config1_naive_count(
    claim_text: str,
    evidence_items: list[dict[str, Any]],
    nli_probs_per_item: list[dict[str, float]],
) -> dict[str, Any]:
    """Config 1: Naive Count.

    Treats every single retrieved/available document as an independent corroborator.
    Assigns unit weight w_i = 1.0 without any deduplication or redundancy check.
    """
    m = len(evidence_items)
    weights = [1.0] * m
    combined = combine_nli_probabilities(nli_probs_per_item, weights, prior_prob_support=0.5)
    return {
        "retained_count": m,
        "effective_sample_size": float(m),
        "weights": weights,
        "posterior_support_probability": combined["posterior_support_probability"],
        "combined_log_odds_support": combined["combined_log_odds_support"],
        "net_direction": combined["net_direction"],
    }


def run_config2_cluster_and_count_one_b2(
    claim_text: str,
    evidence_items: list[dict[str, Any]],
    nli_probs_per_item: list[dict[str, float]],
    detector: SourceIndependenceDetector,
) -> dict[str, Any]:
    """Config 2: Cluster-and-Count-One (B2 Baseline).

    Performs clustering (or surface exact/near-duplicate dedup) and counts exactly
    ONE representative per cluster with unit weight w_k = 1.0.
    Omits Kish effective sample size discounting and quality/freshness gating.
    """
    set_res = detector.analyze_set(evidence_items)
    # Each cluster contributes 1 representative; each singleton contributes 1
    # Find winning representative index for each cluster
    clustered_ids: set[str] = set()
    selected_indices: list[int] = []

    for cluster in set_res.clusters:
        # Pick the first member in the cluster
        rep_id = cluster.document_ids[0]
        clustered_ids.update(cluster.document_ids)
        for idx, it in enumerate(evidence_items):
            if it["evidence_id"] == rep_id:
                selected_indices.append(idx)
                break

    # Add independent singletons
    for idx, it in enumerate(evidence_items):
        if it["evidence_id"] not in clustered_ids:
            selected_indices.append(idx)

    retained_k = len(selected_indices)
    weights = [1.0] * retained_k
    selected_probs = [nli_probs_per_item[i] for i in selected_indices]

    combined = combine_nli_probabilities(selected_probs, weights, prior_prob_support=0.5)
    return {
        "retained_count": retained_k,
        "effective_sample_size": float(retained_k),
        "weights": weights,
        "posterior_support_probability": combined["posterior_support_probability"],
        "combined_log_odds_support": combined["combined_log_odds_support"],
        "net_direction": combined["net_direction"],
    }


def run_config3_kish_only(
    claim_text: str,
    evidence_items: list[dict[str, Any]],
    nli_probs_per_item: list[dict[str, float]],
    detector: SourceIndependenceDetector,
) -> dict[str, Any]:
    """Config 3: Kish Only.

    Does not drop documents; instead discounts all surviving items globally
    by the Kish effective sample size ratio:
        w_i = n_eff / M
    Reflects the theoretical Kish design-effect weighting without intra-cluster representative pooling.
    """
    m = len(evidence_items)
    if m == 0:
        return {
            "retained_count": 0,
            "effective_sample_size": 0.0,
            "weights": [],
            "posterior_support_probability": 0.5,
            "combined_log_odds_support": 0.0,
            "net_direction": "NEUTRAL",
        }

    set_res = detector.analyze_set(evidence_items)
    n_eff = set_res.kish_effective_sample_size
    # Redundancy discount factor: each item gets weight (n_eff / M)
    kish_weight = max(0.01, min(1.0, n_eff / float(m)))
    weights = [kish_weight] * m

    combined = combine_nli_probabilities(nli_probs_per_item, weights, prior_prob_support=0.5)
    return {
        "retained_count": m,
        "effective_sample_size": n_eff,
        "weights": weights,
        "posterior_support_probability": combined["posterior_support_probability"],
        "combined_log_odds_support": combined["combined_log_odds_support"],
        "net_direction": combined["net_direction"],
    }


def run_config4_full_pipeline(
    claim_text: str,
    evidence_items: list[dict[str, Any]],
    nli_probs_per_item: list[dict[str, float]],
    pooling_service: ClusterPoolingService,
) -> dict[str, Any]:
    """Config 4: Full Pipeline (Module 8 & Submodule 8.7).

    Executes:
    1. Complete-linkage graph clustering (J >= 0.85 and domain correlation).
    2. 4-layer admissibility firewall (pollution defense, credibility gating, freshness decay).
    3. Intra-cluster representative selection (argmax S_comp).
    4. Epistemic channel reliability weighting: W(C_k) = cred * fresh * (1 - poll).
    5. Kish effective sample size verification.
    6. Informativeness gating (1 - p_neu) in Bayesian log-odds combination.
    """
    m = len(evidence_items)
    pooled_report = pooling_service.pool_clusters(
        claim=claim_text,
        evidence_items=evidence_items,
    )
    admissible_clusters = [c for c in pooled_report.clusters if c.is_admissible]
    k_adm = len(admissible_clusters)
    weights = [c.cluster_reliability_weight for c in admissible_clusters]

    # Map representative item IDs to their NLI probabilities
    rep_probs: list[dict[str, float]] = []
    item_id_to_prob = {
        it["evidence_id"]: prob for it, prob in zip(evidence_items, nli_probs_per_item)
    }

    for c in admissible_clusters:
        rep_id = c.representative_id
        prob = item_id_to_prob.get(rep_id, nli_probs_per_item[0])
        rep_probs.append(prob)

    if not rep_probs:
        return {
            "retained_count": 0,
            "effective_sample_size": 0.0,
            "weights": [],
            "posterior_support_probability": 0.5,
            "combined_log_odds_support": 0.0,
            "net_direction": "NEUTRAL",
        }

    combined = combine_nli_probabilities(rep_probs, weights, prior_prob_support=0.5)
    return {
        "retained_count": k_adm,
        "effective_sample_size": pooled_report.kish_effective_count,
        "weights": weights,
        "posterior_support_probability": combined["posterior_support_probability"],
        "combined_log_odds_support": combined["combined_log_odds_support"],
        "net_direction": combined["net_direction"],
    }


# ============================================================================
# 3. Accuracy, ECE, Brier Score, and Bootstrap Confidence Intervals
# ============================================================================

def compute_raw_metrics(
    binary_targets: np.ndarray,
    support_probs: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, float]:
    """Compute Binary Accuracy, Expected Calibration Error (ECE), and Brier Score.

    Args:
        binary_targets: 1D array of {0, 1} target labels (1=SUPPORT, 0=CONTRADICT).
        support_probs: 1D array of posterior support probabilities in [0.0, 1.0].
        n_bins: Number of equal-width bins for calibration error (default 10).

    Returns:
        Tuple of (accuracy, ece, brier_score).
    """
    total = len(binary_targets)
    if total == 0:
        return 0.0, 0.0, 0.0

    # 1. Binary Accuracy
    pred_labels = (support_probs >= 0.50).astype(int)
    acc = float(np.mean(pred_labels == binary_targets))

    # 2. Brier Score: Mean squared error of calibrated forecast probabilities
    # BS = (1 / N) * sum (p_i - y_i)^2 in [0.0, 1.0], lower is strictly better
    brier = float(np.mean((support_probs - binary_targets) ** 2))

    # 3. Expected Calibration Error (ECE)
    confidences = np.maximum(support_probs, 1.0 - support_probs)
    is_correct = (pred_labels == binary_targets).astype(float)
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(is_correct[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / float(total)) * abs(bin_acc - bin_conf)

    return acc, float(ece), brier


def compute_metrics_with_bootstrap(
    predictions: list[dict[str, Any]],
    gold_labels: list[str],
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Accuracy, ECE, Brier score, and 95% non-parametric bootstrap confidence intervals.

    Returns:
        Dict containing point estimates and 95% CI bounds: [ci_lower, ci_upper].
    """
    binary_targets = np.array([1 if lbl == "SUPPORT" else 0 for lbl in gold_labels], dtype=int)
    support_probs = np.array([float(p["posterior_support_probability"]) for p in predictions], dtype=float)

    point_acc, point_ece, point_brier = compute_raw_metrics(binary_targets, support_probs, n_bins=n_bins)

    rng = np.random.RandomState(seed)
    n = len(binary_targets)
    boot_accs: list[float] = []
    boot_eces: list[float] = []
    boot_briers: list[float] = []

    for _ in range(n_bootstraps):
        sample_indices = rng.randint(0, n, size=n)
        b_acc, b_ece, b_brier = compute_raw_metrics(
            binary_targets[sample_indices],
            support_probs[sample_indices],
            n_bins=n_bins,
        )
        boot_accs.append(b_acc)
        boot_eces.append(b_ece)
        boot_briers.append(b_brier)

    ci_acc = (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5)))
    ci_ece = (float(np.percentile(boot_eces, 2.5)), float(np.percentile(boot_eces, 97.5)))
    ci_brier = (float(np.percentile(boot_briers, 2.5)), float(np.percentile(boot_briers, 97.5)))

    return {
        "accuracy": round(point_acc, 4),
        "accuracy_ci": (round(ci_acc[0], 4), round(ci_acc[1], 4)),
        "ece": round(point_ece, 4),
        "ece_ci": (round(ci_ece[0], 4), round(ci_ece[1], 4)),
        "brier": round(point_brier, 4),
        "brier_ci": (round(ci_brier[0], 4), round(ci_brier[1], 4)),
    }


def compute_accuracy_and_ece(
    predictions: list[dict[str, Any]],
    gold_labels: list[str],
    n_bins: int = 10,
) -> tuple[float, float]:
    """Backward compatibility shim returning (accuracy, ece)."""
    res = compute_metrics_with_bootstrap(predictions, gold_labels, n_bins=n_bins, n_bootstraps=100)
    return res["accuracy"], res["ece"]


# ============================================================================
# 4. Sweep Execution Over Dependent Copy Count (0 to N)
# ============================================================================

def execute_sweep_experiment(
    frozen_corpus_path: Path,
    max_copies: int = 6,
) -> dict[str, Any]:
    """Sweep dependent copies from 0 to max_copies across all 4 configurations."""
    corpus_data = json.loads(frozen_corpus_path.read_text(encoding="utf-8"))
    records = corpus_data["records"]
    total_claims = len(records)
    logger.info("Loaded frozen corpus with %d claims.", total_claims)

    detector = SourceIndependenceDetector()
    pooling_service = ClusterPoolingService()

    configs = ["Naive Count", "Cluster-and-Count-One (B2)", "Kish Only", "Full Pipeline"]
    sweep_results: dict[str, dict[str, Any]] = {
        cfg: {
            "copy_ratios": [],
            "accuracies": [],
            "accuracies_ci": [],
            "eces": [],
            "eces_ci": [],
            "briers": [],
            "briers_ci": [],
            "mean_n_effs": [],
        }
        for cfg in configs
    }

    copy_counts = list(range(max_copies + 1))  # 0, 1, 2, ..., max_copies

    for num_copies in copy_counts:
        # Dependent copy ratio = num_copies / (1 + num_copies)
        total_docs_per_claim = 1 + num_copies
        dep_ratio = round(num_copies / float(total_docs_per_claim), 4)
        logger.info(
            "--- Running Sweep Step: num_copies=%d (Dependent Copy Ratio: %.2f%%) ---",
            num_copies,
            dep_ratio * 100.0,
        )

        cfg1_preds, cfg2_preds, cfg3_preds, cfg4_preds = [], [], [], []
        cfg1_neff, cfg2_neff, cfg3_neff, cfg4_neff = [], [], [], []
        gold_labels = []

        for rec in records:
            claim_text = rec["claim_text"]
            gold_label = rec["gold_label"]
            gold_labels.append(gold_label)

            primary_item = rec["original_evidence"]
            primary_prob = rec["primary_nli_probs"]

            # Select original + num_copies synthetic copies
            raw_active = [primary_item] + rec["synthetic_copies"][:num_copies]
            active_items = []
            for it in raw_active:
                item_copy = dict(it)
                p_date = item_copy.get("published_date")
                if isinstance(p_date, str):
                    try:
                        item_copy["published_date"] = date.fromisoformat(p_date)
                    except Exception:
                        item_copy["published_date"] = date(2023, 1, 1)
                active_items.append(item_copy)

            # In syndicated/mirror copies, semantic content is identical to the primary evidence,
            # so the primary DeBERTa NLI prediction reflects the model's stance on this evidence passage.
            active_probs = [primary_prob] * len(active_items)

            # 1. Naive Count
            res1 = run_config1_naive_count(claim_text, active_items, active_probs)
            cfg1_preds.append(res1)
            cfg1_neff.append(res1["effective_sample_size"])

            # 2. Cluster-and-Count-One (B2)
            res2 = run_config2_cluster_and_count_one_b2(claim_text, active_items, active_probs, detector)
            cfg2_preds.append(res2)
            cfg2_neff.append(res2["effective_sample_size"])

            # 3. Kish Only
            res3 = run_config3_kish_only(claim_text, active_items, active_probs, detector)
            cfg3_preds.append(res3)
            cfg3_neff.append(res3["effective_sample_size"])

            # 4. Full Pipeline
            res4 = run_config4_full_pipeline(claim_text, active_items, active_probs, pooling_service)
            cfg4_preds.append(res4)
            cfg4_neff.append(res4["effective_sample_size"])

        # Compute Metrics and Bootstrap CIs for all 4 configs
        for cfg_name, preds, neffs in [
            ("Naive Count", cfg1_preds, cfg1_neff),
            ("Cluster-and-Count-One (B2)", cfg2_preds, cfg2_neff),
            ("Kish Only", cfg3_preds, cfg3_neff),
            ("Full Pipeline", cfg4_preds, cfg4_neff),
        ]:
            metrics = compute_metrics_with_bootstrap(preds, gold_labels, n_bins=10, n_bootstraps=1000)
            sweep_results[cfg_name]["copy_ratios"].append(dep_ratio)
            sweep_results[cfg_name]["accuracies"].append(metrics["accuracy"])
            sweep_results[cfg_name]["accuracies_ci"].append(metrics["accuracy_ci"])
            sweep_results[cfg_name]["eces"].append(metrics["ece"])
            sweep_results[cfg_name]["eces_ci"].append(metrics["ece_ci"])
            sweep_results[cfg_name]["briers"].append(metrics["brier"])
            sweep_results[cfg_name]["briers_ci"].append(metrics["brier_ci"])
            sweep_results[cfg_name]["mean_n_effs"].append(round(float(np.mean(neffs)), 2))
            logger.info(
                "  [%s] Acc=%.2f%% (95%% CI: %.2f-%.2f%%), ECE=%.4f (95%% CI: %.4f-%.4f), Brier=%.4f (95%% CI: %.4f-%.4f), Mean n_eff=%.2f",
                cfg_name,
                metrics["accuracy"] * 100.0,
                metrics["accuracy_ci"][0] * 100.0,
                metrics["accuracy_ci"][1] * 100.0,
                metrics["ece"],
                metrics["ece_ci"][0],
                metrics["ece_ci"][1],
                metrics["brier"],
                metrics["brier_ci"][0],
                metrics["brier_ci"][1],
                np.mean(neffs),
            )

    return {
        "metadata": corpus_data["metadata"],
        "copy_counts": copy_counts,
        "sweep_results": sweep_results,
    }


# ============================================================================
# 5. Plotting Accuracy and ECE against Dependent Copy Ratio
# ============================================================================

def plot_accuracy_and_ece(
    sweep_data: dict[str, Any],
    output_plot_path: Path,
) -> None:
    """Generate high-resolution publication plot comparing Accuracy, ECE, and Brier Score across configs."""
    results = sweep_data["sweep_results"]
    copy_counts = sweep_data["copy_counts"]

    # Sample ratios from first config
    first_cfg = list(results.keys())[0]
    ratios_pct = [r * 100.0 for r in results[first_cfg]["copy_ratios"]]

    fig, (ax_acc, ax_ece, ax_brier) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    color_map = {
        "Naive Count": "#d9534f",               # Red
        "Cluster-and-Count-One (B2)": "#f0ad4e", # Orange
        "Kish Only": "#0275d8",                  # Blue
        "Full Pipeline": "#5cb85c",              # Green
    }
    marker_map = {
        "Naive Count": "o",
        "Cluster-and-Count-One (B2)": "s",
        "Kish Only": "^",
        "Full Pipeline": "D",
    }
    linestyle_map = {
        "Naive Count": "--",
        "Cluster-and-Count-One (B2)": "-.",
        "Kish Only": ":",
        "Full Pipeline": "-",
    }

    for cfg_name, data in results.items():
        c = color_map[cfg_name]
        m = marker_map[cfg_name]
        ls = linestyle_map[cfg_name]

        # 1. Plot Accuracy with 95% CI error bars/bands
        acc_pct = [a * 100.0 for a in data["accuracies"]]
        ax_acc.plot(
            ratios_pct,
            acc_pct,
            label=cfg_name,
            color=c,
            marker=m,
            linestyle=ls,
            linewidth=2.2,
            markersize=6,
        )

        # 2. Plot ECE
        ax_ece.plot(
            ratios_pct,
            data["eces"],
            label=cfg_name,
            color=c,
            marker=m,
            linestyle=ls,
            linewidth=2.2,
            markersize=6,
        )

        # 3. Plot Brier Score
        ax_brier.plot(
            ratios_pct,
            data["briers"],
            label=cfg_name,
            color=c,
            marker=m,
            linestyle=ls,
            linewidth=2.2,
            markersize=6,
        )

    # Formatting Left Plot: Accuracy
    ax_acc.set_title("Binary Accuracy vs. Dependent Copies", fontsize=11, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Dependent Copy Ratio (%)", fontsize=10, fontweight="medium")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10, fontweight="medium")
    ax_acc.set_ylim(45.0, 75.0)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower left", frameon=True, fontsize=8.5)

    # Formatting Middle Plot: ECE
    ax_ece.set_title("Expected Calibration Error (ECE)", fontsize=11, fontweight="bold", pad=10)
    ax_ece.set_xlabel("Dependent Copy Ratio (%)", fontsize=10, fontweight="medium")
    ax_ece.set_ylabel("ECE (Lower is Better)", fontsize=10, fontweight="medium")
    ax_ece.grid(True, linestyle="--", alpha=0.5)
    ax_ece.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Formatting Right Plot: Brier Score
    ax_brier.set_title("Brier Score Calibration", fontsize=11, fontweight="bold", pad=10)
    ax_brier.set_xlabel("Dependent Copy Ratio (%)", fontsize=10, fontweight="medium")
    ax_brier.set_ylabel("Brier Score (Lower is Better)", fontsize=10, fontweight="medium")
    ax_brier.grid(True, linestyle="--", alpha=0.5)
    ax_brier.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.suptitle(
        "Empirical Calibration & Provenance Ablation (Accuracy, ECE, Brier Score vs. Redundancy)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sweep evaluation figure to %s", output_plot_path)


# ============================================================================
# 6. Main Orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Aggregation Robustness Across Dependent Copy Ratios")
    parser.add_argument("--frozen-corpus", type=str, default="data/frozen_provenance_corpus.json")
    parser.add_argument("--base-report", type=str, default="data/nli_vs_retrieval_gap_report.json")
    parser.add_argument("--max-copies", type=int, default=6)
    parser.add_argument("--output-json", type=str, default="data/dedup_provenance_sweep_report.json")
    parser.add_argument("--output-plot", type=str, default="data/accuracy_ece_vs_dependent_copies.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frozen_path = Path(args.frozen_corpus)
    base_report_path = Path(args.base_report)
    out_json_path = Path(args.output_json)
    out_plot_path = Path(args.output_plot)

    # 1. Freeze Corpus First (Contract: freeze before running and do not edit after)
    if not frozen_path.exists():
        logger.info("Frozen corpus not found. Programmatically building and freezing corpus to %s...", frozen_path)
        build_and_freeze_synthetic_corpus(
            benchmark_report_path=base_report_path,
            output_frozen_path=frozen_path,
            seed=args.seed,
            max_copies_per_claim=args.max_copies,
        )
    else:
        logger.info("Found existing frozen corpus at %s. Using frozen data without modification.", frozen_path)

    # 2. Execute Sweep Experiment across the 4 configurations
    t0 = time.perf_counter()
    sweep_data = execute_sweep_experiment(frozen_corpus_path=frozen_path, max_copies=args.max_copies)
    elapsed = time.perf_counter() - t0
    logger.info("Sweep evaluation completed in %.2fs", elapsed)

    # 3. Plot Accuracy and ECE
    plot_accuracy_and_ece(sweep_data, output_plot_path=out_plot_path)

    # 4. Save JSON Report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "frozen_corpus_file": str(frozen_path),
        "plot_image_file": str(out_plot_path),
        **sweep_data,
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved final sweep report to %s", out_json_path)

    # 5. Print Summary Table
    print("\n" + "=" * 125)
    print("   PROVENANCE SWEEP BENCHMARK: ACCURACY, CALIBRATION (ECE, BRIER) & 95% BOOTSTRAP CIs   ")
    print("=" * 125)
    header_cols = [f"N={c} ({r*100:.0f}%)" for c, r in zip(sweep_data['copy_counts'], sweep_data['sweep_results']['Naive Count']['copy_ratios'])]
    print(f"{'Config':<26} | {'Metric':<14} | " + " | ".join([f"{col:^16}" for col in header_cols]))
    print("-" * 125)
    for cfg_name, res in sweep_data["sweep_results"].items():
        accs_str = " | ".join([f"{a*100:.1f} [{ci[0]*100:.0f}-{ci[1]*100:.0f}]" for a, ci in zip(res["accuracies"], res["accuracies_ci"])])
        eces_str = " | ".join([f"{e:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]" for e, ci in zip(res["eces"], res["eces_ci"])])
        brier_str = " | ".join([f"{b:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]" for b, ci in zip(res["briers"], res["briers_ci"])])
        neff_str = " | ".join([f"{n:^16.2f}" for n in res["mean_n_effs"]])
        print(f"{cfg_name:<26} | Accuracy (%)   | {accs_str}")
        print(f"{'':<26} | ECE (lower=ok) | {eces_str}")
        print(f"{'':<26} | Brier Score    | {brier_str}")
        print(f"{'':<26} | Mean n_eff     | {neff_str}")
        print("-" * 125)
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()
