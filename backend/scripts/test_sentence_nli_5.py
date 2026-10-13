import json
from pathlib import Path
import sys
import time

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import torch

from app.services.bm25_retriever import BM25Retriever
from app.services.corpus_loader import CorpusLoader
from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine
from app.services.nli_models import NLILabel, SentencePrediction, DocumentNLIPoolResult
from app.services.sentence_window import split_into_sentences

# Load corpus and claims
corpus_loader = CorpusLoader.from_file(Path("data/scifact_corpus.jsonl"))
corpus_map = {doc.id: doc for doc in corpus_loader.documents}
bm25 = BM25Retriever.from_documents(corpus_loader.documents)

claims = []
with open("data/scifact_claims.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("evidence_doc_ids") and d.get("raw_evidence"):
            raw_ev = d["raw_evidence"]
            for doc_key, ev_list in raw_ev.items():
                for ev in ev_list:
                    lbl = ev.get("label", "").upper()
                    if lbl in ("SUPPORT", "CONTRADICT"):
                        d["gold_label"] = lbl
                        claims.append(d)
                        break
                if "gold_label" in d:
                    break

# Select 5 claims (3 SUPPORT, 2 CONTRADICT)
sample_claims = [c for c in claims if c["gold_label"] == "SUPPORT"][:3] + [c for c in claims if c["gold_label"] == "CONTRADICT"][:2]

settings = NLISettings(
    model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device="cpu",
    batch_size=16,
    enable_sentence_windows=False,
)
engine = NLIEngine(settings)

print(f"Testing sentence-level NLI max pooling on {len(sample_claims)} claims...")

for c_idx, claim in enumerate(sample_claims):
    c_text = claim["claim_text"]
    gold_lbl = claim["gold_label"]
    gold_id = claim["evidence_doc_ids"][0]
    gold_doc = corpus_map[gold_id]

    print(f"\n--- Claim {c_idx+1}: [{gold_lbl}] {c_text[:70]} ---")
    
    # 1. Gold Abstract Sentence-Level Scoring
    gold_pool = engine.predict_document_sentences(c_text, gold_doc.text, document_id=gold_id)
    print(f"Gold Abstract: {gold_pool.num_sentences_evaluated} sentences")
    print(f"Max P(ent)={gold_pool.p_entailment_max:.4f}, Max P(con)={gold_pool.p_contradiction_max:.4f}")
    print(f"Predicted Label: {gold_pool.winning_label.value.upper()} (conf: {gold_pool.winning_confidence:.4f})")
    if gold_pool.winning_sentence:
        print(f"Winning sentence [{gold_pool.winning_sentence.sentence_idx}]: {gold_pool.winning_sentence.sentence_text[:90]}...")
    target = "ENTAILMENT" if gold_lbl == "SUPPORT" else "CONTRADICTION"
    print(f"Gold Correct: {gold_pool.winning_label.value.upper() == target}")

    # 2. Retrieved Top-3 Documents Sentence-Level Scoring
    ret_docs = [(r.document_id, r.text) for r in bm25.retrieve(c_text, top_k=3)]
    ret_pool = engine.predict_multi_document_sentences(c_text, ret_docs)
    print(f"Retrieved Top-3: {ret_pool.num_sentences_evaluated} sentences across 3 docs")
    print(f"Max P(ent)={ret_pool.p_entailment_max:.4f}, Max P(con)={ret_pool.p_contradiction_max:.4f}")
    print(f"Predicted Label: {ret_pool.winning_label.value.upper()} (conf: {ret_pool.winning_confidence:.4f})")
    if ret_pool.winning_sentence:
        print(f"Winning sentence [{ret_pool.winning_sentence.doc_id}:{ret_pool.winning_sentence.sentence_idx}]: {ret_pool.winning_sentence.sentence_text[:90]}...")
    print(f"Retrieved Correct: {ret_pool.winning_label.value.upper() == target}")
