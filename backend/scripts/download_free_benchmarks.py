"""Download and normalize real-world benchmark evidence corpora (SciFact & FEVER).

This script streams open-source academic datasets and normalizes them into compact,
memory-safe JSONL files conforming to the CorpusDocument schema:
- SciFact: 5,183 peer-reviewed PubMed biomedical abstracts (AllenAI)
- FEVER: Shared task dev split factual claims with Wikipedia article citations

Zero external cost, zero subscriptions, zero paid API keys.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
import sys
import tarfile
from urllib.parse import quote
from typing import Any

# Ensure backend root is on sys.path when executed directly
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import httpx

from app.config import settings
from app.services.corpus_schema import CorpusDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_free_benchmarks")

SCIFACT_TAR_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"


def normalize_scifact_document(raw_doc: dict[str, Any]) -> CorpusDocument:
    """Transform raw SciFact JSON into a validated CorpusDocument."""
    doc_id = str(raw_doc["doc_id"])
    title = str(raw_doc.get("title", "")).strip() or f"PubMed Abstract {doc_id}"

    # SciFact abstracts are stored as lists of sentence strings
    raw_abstract = raw_doc.get("abstract", [])
    if isinstance(raw_abstract, list):
        text = " ".join(str(s).strip() for s in raw_abstract if str(s).strip())
    else:
        text = str(raw_abstract).strip()

    if not text:
        text = title

    return CorpusDocument(
        id=f"scifact_{doc_id}",
        title=title,
        text=text,
        source="PubMed / AllenAI SciFact",
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/",
        domain="ncbi.nlm.nih.gov",
        published_date=None,
    )


def download_scifact(
    target_dir: Path,
    timeout: float = 45.0,
    skip_if_exists: bool = False,
    max_docs: int | None = None,
) -> tuple[int, int]:
    """Download SciFact tarball, extract corpus and claims, and normalize into JSONL.

    Returns:
        tuple of (documents_written: int, claims_written: int)
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_out_file = target_dir / "scifact_corpus.jsonl"
    claims_out_file = target_dir / "scifact_claims.jsonl"

    if skip_if_exists and corpus_out_file.exists() and claims_out_file.exists():
        logger.info("SciFact files already exist at %s. Skipping download.", target_dir)
        corpus_count = sum(1 for _ in corpus_out_file.open("r", encoding="utf-8"))
        claims_count = sum(1 for _ in claims_out_file.open("r", encoding="utf-8"))
        return corpus_count, claims_count

    logger.info("Downloading SciFact release from %s...", SCIFACT_TAR_URL)
    headers = {"User-Agent": settings.WEB_USER_AGENT}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(SCIFACT_TAR_URL, headers=headers)
        resp.raise_for_status()

    logger.info("Extracting and normalizing SciFact corpus...")
    docs_written = 0
    claims_written = 0

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        # 1. Process corpus.jsonl
        corpus_member = tar.extractfile("data/corpus.jsonl")
        if corpus_member is None:
            raise RuntimeError("data/corpus.jsonl not found in SciFact archive")

        with corpus_out_file.open("w", encoding="utf-8") as f_out:
            for line_bytes in corpus_member:
                line_str = line_bytes.decode("utf-8").strip()
                if not line_str:
                    continue
                raw_doc = json.loads(line_str)
                doc = normalize_scifact_document(raw_doc)
                f_out.write(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False) + "\n")
                docs_written += 1
                if max_docs is not None and docs_written >= max_docs:
                    break

        logger.info("Successfully normalized %d SciFact documents to %s", docs_written, corpus_out_file)

        # 2. Process claims (claims_dev.jsonl + claims_train.jsonl)
        with claims_out_file.open("w", encoding="utf-8") as f_out:
            for split_name in ("data/claims_dev.jsonl", "data/claims_train.jsonl"):
                try:
                    claims_member = tar.extractfile(split_name)
                except KeyError:
                    continue
                if claims_member is None:
                    continue

                for line_bytes in claims_member:
                    line_str = line_bytes.decode("utf-8").strip()
                    if not line_str:
                        continue
                    raw_claim = json.loads(line_str)
                    claim_id = str(raw_claim.get("id"))
                    claim_text = str(raw_claim.get("claim", "")).strip()

                    # Extract cited evidence doc IDs and stance labels
                    evidence_map = raw_claim.get("evidence", {})
                    cited_doc_ids = [f"scifact_{d_id}" for d_id in evidence_map.keys()]

                    normalized_claim = {
                        "claim_id": f"scifact_{claim_id}",
                        "claim_text": claim_text,
                        "split": "dev" if "dev" in split_name else "train",
                        "evidence_doc_ids": cited_doc_ids,
                        "raw_evidence": evidence_map,
                    }
                    f_out.write(json.dumps(normalized_claim, ensure_ascii=False) + "\n")
                    claims_written += 1

        logger.info("Successfully normalized %d SciFact claims to %s", claims_written, claims_out_file)

    return docs_written, claims_written


def download_fever(
    target_dir: Path,
    timeout: float = 45.0,
    skip_if_exists: bool = False,
    max_claims: int | None = 1000,
) -> tuple[int, int]:
    """Stream FEVER shared task dev split and normalize into claims and corpus.

    Returns:
        tuple of (documents_written: int, claims_written: int)
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    claims_out_file = target_dir / "fever_claims.jsonl"
    corpus_out_file = target_dir / "fever_corpus.jsonl"

    if skip_if_exists and claims_out_file.exists() and corpus_out_file.exists():
        logger.info("FEVER files already exist at %s. Skipping download.", target_dir)
        claims_count = sum(1 for _ in claims_out_file.open("r", encoding="utf-8"))
        corpus_count = sum(1 for _ in corpus_out_file.open("r", encoding="utf-8"))
        return corpus_count, claims_count

    logger.info("Streaming FEVER dev split from %s...", FEVER_DEV_URL)
    headers = {"User-Agent": settings.WIKIPEDIA_USER_AGENT}

    claims_written = 0
    unique_wiki_titles: dict[str, str] = {}

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", FEVER_DEV_URL, headers=headers) as resp:
            resp.raise_for_status()

            with claims_out_file.open("w", encoding="utf-8") as f_out:
                for line in resp.iter_lines():
                    line_str = line.strip()
                    if not line_str:
                        continue

                    raw_item = json.loads(line_str)
                    claim_id = str(raw_item.get("id"))
                    claim_text = str(raw_item.get("claim", "")).strip()
                    label = str(raw_item.get("label", "")).strip()

                    # Extract unique Wikipedia titles cited in gold evidence
                    cited_titles: list[str] = []
                    evidence_sets = raw_item.get("evidence", [])
                    for ev_set in evidence_sets:
                        for ev_item in ev_set:
                            # FEVER evidence tuple format: [annotation_id, evidence_id, wikipedia_title, sentence_id]
                            if len(ev_item) >= 3 and ev_item[2]:
                                title = str(ev_item[2]).strip()
                                cited_titles.append(title)
                                if title not in unique_wiki_titles:
                                    clean_display = title.replace("_", " ")
                                    unique_wiki_titles[title] = clean_display

                    normalized_claim = {
                        "claim_id": f"fever_{claim_id}",
                        "claim_text": claim_text,
                        "label": label,
                        "cited_wikipedia_titles": list(dict.fromkeys(cited_titles)),
                    }
                    f_out.write(json.dumps(normalized_claim, ensure_ascii=False) + "\n")
                    claims_written += 1

                    if max_claims is not None and claims_written >= max_claims:
                        break

    logger.info(
        "Successfully normalized %d FEVER claims with %d cited Wikipedia articles.",
        claims_written,
        len(unique_wiki_titles),
    )

    # 2. Build reference corpus documents for cited Wikipedia entities
    docs_written = 0
    with corpus_out_file.open("w", encoding="utf-8") as f_out:
        for wiki_slug, display_title in unique_wiki_titles.items():
            doc = CorpusDocument(
                id=f"fever_wiki_{wiki_slug.lower()}",
                title=display_title,
                text=f"Wikipedia encyclopedia article for '{display_title}'. Reference entity in FEVER validation benchmark.",
                source="FEVER / Wikipedia",
                source_url=f"https://en.wikipedia.org/wiki/{quote(wiki_slug)}",
                domain="en.wikipedia.org",
                published_date=None,
            )
            f_out.write(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False) + "\n")
            docs_written += 1

    logger.info("Successfully normalized %d FEVER reference documents to %s", docs_written, corpus_out_file)

    return docs_written, claims_written


def main() -> None:
    """CLI entry point for downloading academic benchmark datasets."""
    parser = argparse.ArgumentParser(
        description="Download and normalize free benchmark evidence corpora (SciFact & FEVER)."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save normalized JSONL corpora files (default: data/)",
    )
    parser.add_argument(
        "--corpora",
        choices=["all", "scifact", "fever"],
        default="all",
        help="Which benchmark corpora to download (default: all)",
    )
    parser.add_argument(
        "--max-fever-claims",
        type=int,
        default=500,
        help="Maximum FEVER claims to process (default: 500; use 0 for unlimited)",
    )
    parser.add_argument(
        "--max-scifact-docs",
        type=int,
        default=None,
        help="Maximum SciFact documents to process (default: None, processes all 5,183)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Network request timeout in seconds (default: 45.0)",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip downloading if output files already exist",
    )

    args = parser.parse_args()
    max_fever = None if args.max_fever_claims <= 0 else args.max_fever_claims

    print("\n=======================================================")
    print(" FREE BENCHMARK EVIDENCE CORPORA DOWNLOADER")
    print("=======================================================")
    print(f"Target Directory: {args.target_dir.resolve()}")
    print(f"Corpora:          {args.corpora}")
    print(f"Max FEVER Claims: {max_fever or 'ALL'}")
    print(f"Skip if Exists:   {args.skip_if_exists}")
    print("-------------------------------------------------------\n")

    if args.corpora in ("all", "scifact"):
        print("[1/2] Processing SciFact (AllenAI PubMed Corpus)...")
        docs, claims = download_scifact(
            target_dir=args.target_dir,
            timeout=args.timeout,
            skip_if_exists=args.skip_if_exists,
            max_docs=args.max_scifact_docs,
        )
        print(f"      Done: {docs} documents, {claims} claims.\n")

    if args.corpora in ("all", "fever"):
        print("[2/2] Processing FEVER (Fact Extraction and VERification)...")
        docs, claims = download_fever(
            target_dir=args.target_dir,
            timeout=args.timeout,
            skip_if_exists=args.skip_if_exists,
            max_claims=max_fever,
        )
        print(f"      Done: {docs} reference documents, {claims} claims.\n")

    print("=======================================================")
    print(" BENCHMARK DOWNLOAD & NORMALIZATION COMPLETE")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
