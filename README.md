# RAG over the Constitution of Nepal (2015)

A Retrieval-Augmented Generation (RAG) system that answers questions about the Constitution of Nepal, grounded in the actual document text. This project's real value isn't the final working pipeline — it's the debugging journey: three separate retrieval failures were diagnosed and fixed, each revealing a genuine, well-known RAG limitation.

## Data
- Source: Constitution of Nepal 2015 (unofficial English translation, as amended through 2020 — the current legally in-force version as of this project). Note: a further constitutional amendment was under discussion in the Nepali government as of 2026, which may not be reflected in this text.
- 161 pages, ~340,000 characters, 308 numbered articles (285 detected via structure-aware parsing)

## Pipeline
1. **Extract** text from the PDF (`pypdf`)
2. **Chunk** the text (see debugging journey below for how this evolved)
3. **Embed** chunks using `sentence-transformers` (`all-MiniLM-L6-v2`)
4. **Retrieve** relevant chunks via hybrid search (BM25 keyword + FAISS embedding similarity)
5. **Generate** an answer using Google's Gemini API, grounded strictly in retrieved context

## The debugging journey

### Failure 1: Fixed-size chunking cuts sentences mid-way
Initial approach: split text into 500-character chunks with 50-character overlap. Testing "How is the Prime Minister of Nepal appointed?" returned an incomplete answer — the retrieved chunk literally cut off mid-sentence ("The President shall appoint the parliamentary party l...").

**Fix attempted:** increased chunk size to 1000 characters, overlap to 200. This fixed the PM appointment question.

### Failure 2: Larger chunks dilute short facts
The larger-chunk fix broke a previously-working query: "What does the constitution say about the death penalty?" returned "no information," even though the exact text ("No law shall be made for capital punishment") was confirmed to exist via direct string search.

**Diagnosis:** the relevant clause is a single short sentence embedded inside a much longer chunk about "Right to Freedom" (movement, opinion, assembly). The chunk's overall embedding was dominated by its longer, unrelated content, diluting the short but important fact. Verified via direct cosine similarity comparison between the query and the chunk (0.297) — low regardless of query phrasing, confirming dilution rather than a vocabulary mismatch.

**Fix:** structure-aware chunking — split by the document's own numbered articles (regex-matched "N. Title:" patterns) instead of fixed character counts, so each chunk is one complete, atomic legal provision (285 chunks). This fixed both the PM appointment and death penalty queries simultaneously.

### Failure 3: Semantic search misses framing mismatches
Even with article-based chunking, "How is the Prime Minister of Nepal appointed?" still failed. Direct rank inspection showed Article 76 ("Formation of the Council of Ministers") — which fully answers the question — ranked **49th out of 285** chunks for that query.

**Diagnosis:** the article's title uses "Council of Ministers," not "Prime Minister" or "appointment" — the embedding model weighs this framing heavily, and the semantic distance between "Council of Ministers" and a question about "Prime Minister... appointed" was larger than expected, even though the content directly answers the question.

**Fix:** hybrid search — combine BM25 keyword search (which matches "Prime Minister" wherever it literally appears in the text, regardless of the article's title) with embedding-based semantic search, then merge normalized scores. This retrieved Article 76 correctly, plus two additional relevant articles (Article 100 — no-confidence motion scenario, Article 298 — transitional provision) that pure embedding search never surfaced even at high k.

## Key takeaway

RAG retrieval quality depends heavily on the interaction between chunking strategy and document structure, and on whether a query's vocabulary overlaps with the source document's own terminology. No single fix solved everything — each stage revealed a different, genuine failure mode:
- Fixed-size chunking → truncation
- Larger chunking → dilution of short facts in long chunks
- Pure semantic search → misses terminology/framing mismatches between questions and formal/legal document titles

Hybrid search (keyword + embedding) combined with structure-aware chunking resolved all three, though this required actually diagnosing each failure with direct evidence (string search, similarity scores, rank inspection) rather than guessing at fixes.

## Structure
```
src/
  ingest.py     # PDF text extraction, structure-aware article chunking
  retrieve.py   # HybridRetriever class (BM25 + FAISS embedding search)
  generate.py   # Prompt construction and Gemini API call
notebooks/      # step-by-step exploration notebook (Colab), with markdown explanations
```

## Stack
Python, `sentence-transformers`, FAISS, `rank_bm25`, Google Gemini API, `pypdf`

## What's next
- Confirm results hold across a broader question set (paused due to free-tier API daily quota; revisit with a fresh quota or a different provider)
- Query rewriting (have an LLM rephrase informal questions into constitution-style terminology before retrieval) as a further fix for framing mismatches
- Extend to Nepali-language questions, given the source is an English translation of a Nepali original.