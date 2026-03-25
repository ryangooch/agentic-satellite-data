# agent/rag.py
"""Local RAG (Retrieval-Augmented Generation) for agricultural reference documents.

Provides context from county ag reports, extension service bulletins, and
spectral index references. Uses simple TF-IDF similarity for retrieval —
no external vector DB or embedding API needed.

Documents live in data/rag_documents/ as markdown files.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

_DOCS_DIR = Path("data/rag_documents")

# Module-level index (built on first query)
_INDEX: Optional["RAGIndex"] = None


@dataclass
class RAGResult:
    success: bool
    chunks: Optional[list[dict]] = None
    summary: Optional[str] = None
    error_message: Optional[str] = None


class RAGIndex:
    """Simple TF-IDF index over chunked markdown documents."""

    def __init__(self, docs_dir: Path = _DOCS_DIR, chunk_size: int = 500, chunk_overlap: int = 100):
        self.docs_dir = docs_dir
        self.chunks: list[dict] = []  # {"text": str, "source": str, "section": str}
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.tfidf_matrix: Optional[csr_matrix] = None
        self._build_index(chunk_size, chunk_overlap)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def _chunk_document(self, text: str, source: str, chunk_size: int, overlap: int) -> list[dict]:
        """Split a document into overlapping chunks, preserving section context."""
        chunks = []
        current_section = ""

        # Split by markdown headers to preserve section context
        lines = text.split("\n")
        paragraphs = []
        current_para = []

        for line in lines:
            if line.startswith("#"):
                if current_para:
                    paragraphs.append(("\n".join(current_para), current_section))
                    current_para = []
                current_section = line.strip("# ").strip()
                current_para.append(line)
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append(("\n".join(current_para), current_section))

        # Now chunk the paragraphs
        buffer = ""
        buffer_section = ""
        for para_text, section in paragraphs:
            if len(buffer) + len(para_text) > chunk_size and buffer:
                chunks.append({
                    "text": buffer.strip(),
                    "source": source,
                    "section": buffer_section,
                })
                # Keep overlap
                words = buffer.split()
                overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
                buffer = " ".join(overlap_words) + "\n" + para_text
                buffer_section = section
            else:
                buffer += "\n" + para_text
                if not buffer_section:
                    buffer_section = section

        if buffer.strip():
            chunks.append({
                "text": buffer.strip(),
                "source": source,
                "section": buffer_section,
            })

        return chunks

    def _build_index(self, chunk_size: int, chunk_overlap: int):
        """Build the TF-IDF index from all documents in the docs directory."""
        if not self.docs_dir.exists():
            return

        # Load and chunk all documents
        for doc_path in sorted(self.docs_dir.glob("*.md")):
            text = doc_path.read_text()
            source = doc_path.stem.replace("_", " ").title()
            doc_chunks = self._chunk_document(text, source, chunk_size, chunk_overlap)
            self.chunks.extend(doc_chunks)

        if not self.chunks:
            return

        # Build vocabulary
        all_tokens = [self._tokenize(c["text"]) for c in self.chunks]
        vocab_set: dict[str, int] = {}
        for tokens in all_tokens:
            for t in tokens:
                if t not in vocab_set:
                    vocab_set[t] = len(vocab_set)
        self.vocab = vocab_set

        n_docs = len(self.chunks)
        n_vocab = len(self.vocab)

        # Compute TF-IDF
        # TF: term frequency per document
        rows, cols, data = [], [], []
        doc_freq = np.zeros(n_vocab)

        for doc_idx, tokens in enumerate(all_tokens):
            tf = {}
            for t in tokens:
                tid = self.vocab[t]
                tf[tid] = tf.get(tid, 0) + 1
            for tid, count in tf.items():
                rows.append(doc_idx)
                cols.append(tid)
                data.append(1 + np.log(count))  # log-scaled TF
                doc_freq[tid] += 1

        # IDF
        self.idf = np.log(n_docs / (doc_freq + 1)) + 1

        # Build sparse TF-IDF matrix
        tf_matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_vocab))
        # Multiply by IDF
        self.tfidf_matrix = tf_matrix.multiply(self.idf)

        # Normalize rows
        norms = np.sqrt(self.tfidf_matrix.multiply(self.tfidf_matrix).sum(axis=1))
        norms = np.array(norms).flatten()
        norms[norms == 0] = 1
        self.tfidf_matrix = self.tfidf_matrix.multiply(1 / norms[:, np.newaxis])

    def query(self, question: str, top_k: int = 3) -> list[dict]:
        """Find the most relevant chunks for a question."""
        if self.tfidf_matrix is None or len(self.chunks) == 0:
            return []

        # Vectorize the query
        tokens = self._tokenize(question)
        q_vec = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                q_vec[self.vocab[t]] = 1
        q_vec *= self.idf

        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec /= norm

        # Cosine similarity
        scores = self.tfidf_matrix.dot(q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:  # minimum relevance threshold
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "section": self.chunks[idx]["section"],
                    "score": round(float(scores[idx]), 3),
                })
        return results


def _get_index() -> RAGIndex:
    """Lazy-initialize the RAG index."""
    global _INDEX
    if _INDEX is None:
        _INDEX = RAGIndex()
    return _INDEX


def search_agricultural_context(query: str, top_k: int = 3) -> RAGResult:
    """Search local agricultural reference documents for context relevant to a query.

    Searches county crop reports, UCCE extension bulletins, water district advisories,
    and spectral index reference guides. Returns the most relevant text passages.

    Args:
        query: Natural language question (e.g., "what causes low NDVI in almonds in July?")
        top_k: Number of passages to return (default 3)
    """
    try:
        index = _get_index()
        chunks = index.query(query, top_k=top_k)
    except Exception as e:
        return RAGResult(success=False, error_message=str(e))

    if not chunks:
        return RAGResult(
            success=True,
            chunks=[],
            summary="No relevant documents found for this query.",
        )

    lines = [f"Found {len(chunks)} relevant passage(s):\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"--- Passage {i} (score: {chunk['score']}) ---")
        lines.append(f"Source: {chunk['source']}")
        if chunk["section"]:
            lines.append(f"Section: {chunk['section']}")
        lines.append(chunk["text"][:600])
        if len(chunk["text"]) > 600:
            lines.append("... [truncated]")
        lines.append("")

    return RAGResult(success=True, chunks=chunks, summary="\n".join(lines))


if __name__ == "__main__":
    rag_index = _get_index()

    print(rag_index.chunks[0])

    results: RAGResult = search_agricultural_context(query="alfalfa coefficient")

    for res in results.chunks:
        print(res)
