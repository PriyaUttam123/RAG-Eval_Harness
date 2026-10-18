# Evaluation Datasets: Golden and Holdout Structure

This document details the organization, schema, and curation principles for the evaluation datasets used in the RAG Evaluation Harness.

---

## 1. Purpose of Dual Datasets

To ensure rigorous and reproducible benchmark results without hyperparameter overfitting, the evaluation architecture maintains two separate, non-overlapping datasets:

### Golden Dataset (`data/evaluation/golden.json`)
- **Purpose**: Used for iterative development, diagnostic analysis, and hyperparameter tuning (e.g., experimenting with chunk sizes, chunk overlap, embedding models, and top-k retrieval depth).
- **Usage**: Engineers inspect retrieval failures and generation scores against this dataset to guide pipeline improvements.

### Holdout Dataset (`data/evaluation/holdout.json`)
- **Purpose**: Serves as a blind validation benchmark to guard against overfitting to the golden dataset.
- **Usage**: Kept separate and evaluated to verify that configuration tuning generalizes to unseen queries.

---

## 2. Dataset Separation & Anti-Leakage Policy

- **No Overlap**: Questions in the golden dataset must **never** appear in the holdout dataset (enforced programmatically by `validate_separation()`, which checks normalized casing and whitespace).
- **Grounded in Actual Corpus**: All benchmark questions and expected answers must be derived from actual ingested source documents. Do not synthesize questions about hypothetical content that does not exist in the corpus.
- **No Fabricated Benchmarks**: Benchmark cases must reflect realistic user inquiries with verified ground-truth facts. Do not invent benchmark results or populate datasets with fabricated dummy cases.

---

## 3. Expected JSON Structure

Datasets are stored as structured JSON files adhering to the following schema:

```json
{
  "cases": [
    {
      "question": "What is the return policy window?",
      "expected_answer": "Customers may request a refund within 30 days of purchase with proof of purchase.",
      "relevant_chunk_ids": ["chunk-policy-001", "chunk-policy-002"],
      "case_type": "factual",
      "metadata": {
        "source_document": "refund_policy.pdf",
        "difficulty": "easy"
      }
    }
  ]
}
```

### Field Specifications

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | `string` | Yes | The query string (must be non-empty). |
| `expected_answer` | `string` | Yes | Ground-truth reference answer used for answer quality evaluation. |
| `relevant_chunk_ids` | `list[string]` | No (default `[]`) | Identifiers of supporting chunks that contain the evidence needed to answer the question. |
| `case_type` | `string` | Yes | Category of evaluation inquiry (see supported types below). |
| `metadata` | `object` | No (default `{}`) | Optional arbitrary metadata (e.g. source filename, difficulty level). |

---

## 4. Supported Case Types

1. **`factual`**:
   - Queries with clear, direct supporting evidence within one or more document chunks.
   - Requires non-empty `relevant_chunk_ids` pointing to supporting chunks.
2. **`tricky`**:
   - Inquiries that involve nuanced distinctions, potential distractor chunks, edge cases, or multi-faceted conditions.
   - Evaluates whether the retriever avoids irrelevant distractors and surfaces exact supporting evidence.
3. **`unanswerable`**:
   - Inquiries about topics not covered by the ingested corpus.
   - Evaluates whether the system appropriately refrains from hallucinating and triggers grounded refusal behavior (e.g., *"I don't have enough information in the provided context"*).
   - **Important**: `relevant_chunk_ids` is intentionally empty (`[]`) for unanswerable cases.

---

## 5. Current Dataset Status

- Initial dataset files (`data/evaluation/golden.json` and `data/evaluation/holdout.json`) are initialized with empty valid schemas (`{"cases": []}`).
- Grounded benchmark cases will be curated directly from ingested domain documents in subsequent milestones.
