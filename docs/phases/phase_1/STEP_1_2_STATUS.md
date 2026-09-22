# Step 1.2 Status: Reference Corpus & Indexing Pipeline

## Overview
Phase 1, Step 1.2 implemented the typed corpus schema, document provenance metadata, corpus loading utilities, and a 15-document reference dataset covering science, history, geopolitics, and technology.

## Architecture & Schemas
- **`CorpusDocument`**:
  - `id`: Unique string identifier
  - `title`: Document title
  - `text`: Primary factual content
  - `source`: Provenance source name
  - `source_url`: Optional source URL
  - `domain`: Optional thematic domain
  - `published_date`: Optional ISO date
- **`Corpus`**:
  - Validates uniqueness of document IDs (`DuplicateDocumentIDError`).
  - Preserves duplicate or near-duplicate texts across different IDs (`test_duplicate_text_preserved`).
- **`CorpusLoader`**:
  - `from_file(path)`
  - `from_json(json_str)`
  - `from_dict(dict_obj)`

## Test Results
- `tests/test_corpus_schema.py`: 12 tests passed.
- `tests/test_corpus_loader.py`: 9 tests passed.
- Total: 21/21 passing.
