import logfire
from app.config import settings

# =============================================================================
# DEMO MODE: Local embeddings (sentence-transformers) instead of Vertex AI.
#
# WHY: Vertex AI requires active GCP billing. This project is built for
# GCP-native production deployment, but for local demo/dev purposes
# (no billing account active yet), we swap in a free, local embedding
# model that outputs the same 768 dimensions as Vertex AI's
# text-embedding-004, so nothing else in the pipeline (Qdrant collection,
# reranker, etc.) needs to change.
#
# TO SWITCH BACK TO VERTEX AI (production): set USE_LOCAL_EMBEDDINGS=False
# below, or make it an env var, and uncomment the Vertex AI block.
# =============================================================================
USE_LOCAL_EMBEDDINGS = True

model = None
BATCH_SIZE = 50


def get_embedding_model():
    global model
    if model is None:
        if USE_LOCAL_EMBEDDINGS:
            from sentence_transformers import SentenceTransformer
            logfire.info("🧩 Loading LOCAL embedding model (all-mpnet-base-v2, 768-dim) — demo mode, no GCP billing required.")
            model = SentenceTransformer("all-mpnet-base-v2")
        else:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
            vertexai.init(project=settings.PROJECT_ID, location=settings.LOCATION)
            model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return model


def embed_query(query: str):
    """Embeds a single query string."""
    m = get_embedding_model()
    if USE_LOCAL_EMBEDDINGS:
        return m.encode(query).tolist()
    else:
        embeddings = m.get_embeddings([query])
        return embeddings[0].values


def embed_texts(texts: list[str]):
    """Embeds a list of text strings in batches."""
    m = get_embedding_model()
    all_embeddings = []

    if USE_LOCAL_EMBEDDINGS:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            vectors = m.encode(batch)
            all_embeddings.extend([v.tolist() for v in vectors])
        return all_embeddings

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = m.get_embeddings(batch)
        all_embeddings.extend([e.values for e in embeddings])

    return all_embeddings