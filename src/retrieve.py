import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(self, chunks, embed_model_name='all-MiniLM-L6-v2'):
        self.chunks = chunks
        self.embed_model = SentenceTransformer(embed_model_name)

        embeddings = self.embed_model.encode(chunks, show_progress_bar=True)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        tokenized_chunks = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, k=5, bm25_weight=0.5):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores_norm = bm25_scores / (bm25_scores.max() + 1e-8)

        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k=len(self.chunks)
        )
        embed_scores = np.zeros(len(self.chunks))
        max_dist = distances[0].max()
        for rank, idx in enumerate(indices[0]):
            embed_scores[idx] = 1 - (distances[0][rank] / (max_dist + 1e-8))

        combined_scores = bm25_weight * bm25_scores_norm + (1 - bm25_weight) * embed_scores
        top_k_idx = np.argsort(combined_scores)[::-1][:k]

        return [self.chunks[i] for i in top_k_idx]