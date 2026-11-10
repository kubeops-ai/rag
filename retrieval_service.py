"""HTTP service that keeps the RAG retrieval pipeline available."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieval import retrieve_context, setup_logging


app = FastAPI(title="KubeOps RAG Retrieval Service")


class RetrievalRequest(BaseModel):
    """Question sent by the KubeOps agent."""

    question: str


class RetrievalResponse(BaseModel):
    """Relevant context returned to the KubeOps agent."""

    context: str


@app.get("/health")
def health_check():
    """Confirm that the retrieval service is running."""
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest):
    """Retrieve RAG context for one question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        context = retrieve_context(request.question)
    except Exception as error:
        print(f"RAG retrieval failed: {error}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="RAG retrieval is temporarily unavailable.",
        ) from error

    return RetrievalResponse(context=context)


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run("retrieval_service:app", host="127.0.0.1", port=8100)
