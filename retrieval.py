"""RAG retrieval pipeline.

Step 1 currently accepts the user's question. Retrieval logic will be added
one step at a time.
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


logger = logging.getLogger("kubeops.retrieval")
EMBEDDING_MODEL = "text-embedding-3-small"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "kubeops_documents"
TOP_K = 10
SCORE_THRESHOLD = 0.3
RERANK_MODEL = "gpt-5.6-luna"
RERANK_LIMIT = 5
MAX_CONTEXT_TOKENS = 2000
RERANK_SYSTEM_PROMPT = (
    "You are a Kubernetes troubleshooting document reranker. "
    "Rank only the supplied chunks for relevance to the user question. "
    "Do not invent chunk IDs, facts, or explanations."
)
COMPRESSION_SYSTEM_PROMPT = (
    "You are a technical documentation editor. "
    "Shorten Kubernetes context while preserving facts, commands, causes, "
    "and remediation steps. Do not invent or change technical information."
)


def get_user_question(question):
    """Return the question received from the calling application."""
    logger.info("Step 1: received the user question.")
    return question.strip()


def rewrite_question(question):
    """Step 2: make small local improvements when the question needs them."""
    # Replace line breaks and repeated spaces without changing the meaning.
    rewritten_question = " ".join(question.split())

    if rewritten_question != question:
        logger.info("Step 2: question whitespace was cleaned.")
    else:
        logger.info("Step 2: no question rewrite was needed.")

    return rewritten_question


def create_openai_client():
    """Create an OpenAI client using OPENAI_API_KEY."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before creating the question embedding.")
    return OpenAI(api_key=api_key)


def create_question_embedding(openai_client, question):
    """Step 3: convert the rewritten question into a vector."""
    logger.info("Step 3: creating the question embedding.")
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )
    question_vector = response.data[0].embedding
    logger.info("Question embedding created. Vector size: %d.", len(question_vector))
    return question_vector


def search_qdrant(question_vector, top_k=TOP_K, metadata_filters=None):
    """Step 4: find relevant chunks above the similarity threshold."""
    logger.info(
        "Step 4: searching Qdrant for the top %d chunks with score threshold %.2f.",
        top_k,
        SCORE_THRESHOLD,
    )
    qdrant = QdrantClient(url=QDRANT_URL)
    collection_count = qdrant.count(
        collection_name=QDRANT_COLLECTION,
        exact=True,
    ).count
    print(
        f"RAG Qdrant collection '{QDRANT_COLLECTION}' contains "
        f"{collection_count} point(s)",
        flush=True,
    )
    search_filter = None

    # Build filters only when the caller provides them.
    if metadata_filters:
        conditions = []
        for field_name, field_value in metadata_filters.items():
            conditions.append(
                FieldCondition(
                    key=field_name,
                    match=MatchValue(value=field_value),
                )
            )
        search_filter = Filter(must=conditions)

    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=question_vector,
        limit=top_k,
        score_threshold=SCORE_THRESHOLD,
        query_filter=search_filter,
        with_payload=True,
        with_vectors=False,
    )
    points = results.points
    print(f"RAG Qdrant hits: {len(points)}", flush=True)
    if not points and collection_count:
        closest = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=question_vector,
            limit=3,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False,
        ).points
        print("No points passed the threshold. Closest scores were:", flush=True)
        for point in closest:
            payload = point.payload or {}
            print(
                f"  source={payload.get('file_name', 'Unknown')} | "
                f"heading={payload.get('heading', 'Unknown')} | score={point.score:.4f}",
                flush=True,
            )
    for number, point in enumerate(points, start=1):
        payload = point.payload or {}
        print(
            f"RAG hit {number}: source={payload.get('file_name', 'Unknown')} | "
            f"heading={payload.get('heading', 'Unknown')} | score={point.score:.4f}",
            flush=True,
        )
    return points


def rerank_results(openai_client, question, results):
    """Step 5: ask the model to reorder the retrieved chunks."""
    if not results:
        return []

    candidates = []
    for result in results:
        payload = result.payload or {}
        candidates.append(
            {
                "chunk_id": str(result.id),
                "heading": payload.get("heading", "Unknown"),
                "content": payload.get("content", ""),
            }
        )

    prompt = (
        "Rank these troubleshooting chunks for the user question. Return only "
        'JSON in this format: {"ranked_chunk_ids": ["id1", "id2"]}. '
        "Put the most useful chunk first. Do not create new IDs.\n\n"
        f"User question:\n{question}\n\n"
        f"Candidate chunks:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    logger.info(
        "Step 5: reranking %d retrieved chunk(s) with %s.",
        len(results),
        RERANK_MODEL,
    )
    try:
        response = openai_client.responses.create(
            model=RERANK_MODEL,
            input=[
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as error:
        logger.warning("Reranking failed; keeping Qdrant order: %s", error)
        return results[:RERANK_LIMIT]

    try:
        ranked_ids = json.loads(response.output_text)["ranked_chunk_ids"]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Reranker response was invalid; keeping Qdrant order.")
        return results[:RERANK_LIMIT]

    results_by_id = {str(result.id): result for result in results}
    ranked_results = [
        results_by_id[chunk_id]
        for chunk_id in ranked_ids
        if chunk_id in results_by_id
    ]
    return ranked_results[:RERANK_LIMIT]


def construct_context(results, max_tokens=MAX_CONTEXT_TOKENS):
    """Step 6: combine reranked chunks into one context string."""
    logger.info("Step 6: constructing context from %d chunk(s).", len(results))
    context_parts = []

    for number, result in enumerate(results, start=1):
        payload = result.payload or {}
        source = payload.get("file_name", "Unknown")
        heading = payload.get("heading", "Unknown")
        content = payload.get("content", "")
        context_part = (
            f"[Context {number}]\n"
            f"Source: {source}\n"
            f"Heading: {heading}\n"
            f"Content:\n{content}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)


def fit_context_with_llm(openai_client, question, context):
    """Step 7: ask the LLM to fit context within the token budget."""
    estimated_tokens = len(context) // 4
    print(
        f"Context token estimate: {estimated_tokens} / "
        f"{MAX_CONTEXT_TOKENS} budget"
    )
    if estimated_tokens <= MAX_CONTEXT_TOKENS:
        print("Context fits within the token budget.")
        return context

    print("Context is above the token budget; asking the LLM to shorten it.")
    prompt = (
        f"Shorten the following Kubernetes troubleshooting context to fit within "
        f"{MAX_CONTEXT_TOKENS} tokens. Preserve all facts, commands, causes, and "
        "remediation steps. Remove repetition only. Return only the shortened context.\n\n"
        f"User question:\n{question}\n\n"
        f"Context:\n{context}"
    )
    try:
        response = openai_client.responses.create(
            model=RERANK_MODEL,
            input=[
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=MAX_CONTEXT_TOKENS,
        )
    except Exception as error:
        logger.warning("Context compression failed; keeping original context: %s", error)
        return context
    print("Context was shortened by the LLM.")
    return response.output_text


def retrieve_context(question, metadata_filters=None):
    """Run retrieval and return context for a calling application."""
    question = get_user_question(question)
    rewritten_question = rewrite_question(question)
    openai_client = create_openai_client()
    question_vector = create_question_embedding(openai_client, rewritten_question)
    results = search_qdrant(question_vector, metadata_filters=metadata_filters)
    reranked_results = rerank_results(openai_client, rewritten_question, results)
    context = construct_context(reranked_results)
    final_context = fit_context_with_llm(openai_client, rewritten_question, context)
    print("RAG context returned to KubeOps agent:", flush=True)
    print(final_context or "No matching RAG context found.", flush=True)
    return final_context


def setup_logging():
    """Configure simple logs for the retrieval steps."""
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )


if __name__ == "__main__":
    setup_logging()
    print(
        "Retrieval is a library function. Run retrieval_service.py to start "
        "the background retrieval service."
    )
