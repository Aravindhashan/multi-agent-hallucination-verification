import sys
import json
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.verification_service import get_verification_service

service = get_verification_service()

test_text = (
    "Autophagy deficiency leads to the accumulation of damaged mitochondria. "
    "MicroRNA-21 expression is downregulated in colorectal cancer tissues."
)

print("Executing complete verification pipeline across claims...")
resp = service.verify_text(text=test_text, evidence_mode="benchmark_scifact")

print("\n" + "=" * 90)
print("             COMPLETE PIPELINE END-TO-END VERIFICATION AUDIT")
print("=" * 90)
print(f"Overall Statement Verdict    : {resp.overall_verdict.value}")
print(f"Factual Hallucination Risk H : {resp.hallucination_index:.4f} (0.0 = Fully Factual, 1.0 = Pure Refuted Fabrication)")
print(f"Epistemic Uncertainty Index U: {resp.unverified_index:.4f} (0.0 = Decisive Evidence, 1.0 = Inconclusive/Missing)")
print(f"Calibrated Confidence Score  : {resp.confidence_score * 100:.2f}%")
print(f"Total Atomic Claims Evaluated: {resp.claim_count}")
print(f"Supported Claims Count       : {resp.supported_count}")
print(f"Refuted (Opposed) Claims     : {resp.refuted_count}")
print(f"Unverified (NEI) Claims      : {resp.unverified_count}")
print(f"Aggregate Joint Kish n_eff   : {resp.aggregate_joint_n_eff:.2f}")
print(f"Execution Latency            : {resp.execution_time_ms:.1f} ms")
print("-" * 90)

for idx, c in enumerate(resp.claims, 1):
    print(f"\n[CLAIM {idx}] \"{c.claim_text}\"")
    print(f"  * Individual Verdict        : {c.verdict.value}")
    print(f"  * Calibrated Confidence     : {c.confidence_score * 100:.2f}%")
    print(f"  * Support Probability       : {c.support_probability * 100:.2f}%")
    print(f"  * Contradiction Probability : {c.contradiction_probability * 100:.2f}%")
    print(f"  * Neutral Probability       : {c.neutral_probability * 100:.2f}%")
    print(f"  * Kish n_eff (Supp / Opp)   : {c.supporting_n_eff:.2f} / {c.contradicting_n_eff:.2f}")
    
    print(f"\n  >>> SUPPORTING CLAIMS & EVIDENCE ({len(c.supportive_claims)} retrieved):")
    if not c.supportive_claims:
        print("      (None found)")
    for s_idx, s in enumerate(c.supportive_claims, 1):
        p_ent = f"{s.p_entailment * 100:.2f}%" if s.p_entailment is not None else "N/A"
        src = s.domain or s.source_url or "scifact_corpus"
        print(f"      [{s_idx}] Entailment Probability: {p_ent} | Source: {src}")
        clean_text = " ".join(s.text.split())
        print(f"          Snippet: {clean_text[:140]}...")
        
    print(f"\n  >>> OPPOSING CLAIMS & EVIDENCE ({len(c.opposing_claims)} retrieved):")
    if not c.opposing_claims:
        print("      (None found)")
    for o_idx, o in enumerate(c.opposing_claims, 1):
        p_cont = f"{o.p_contradiction * 100:.2f}%" if o.p_contradiction is not None else "N/A"
        src = o.domain or o.source_url or "scifact_corpus"
        print(f"      [{o_idx}] Contradiction Probability: {p_cont} | Source: {src}")
        clean_text = " ".join(o.text.split())
        print(f"          Snippet: {clean_text[:140]}...")
    print("-" * 90)

print("=" * 90)
