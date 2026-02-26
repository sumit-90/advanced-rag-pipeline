from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    EvaluateRequest, EvaluateResponse
)
from src.pipeline import run_ingestion_pipeline, run_query_pipeline
from src.evaluation.evaluator import evaluate_pipeline
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="RAG Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    logger.info(f"Ingest request received for directory: {request.directory}")
    result = run_ingestion_pipeline(request.directory)
    if result['status'] == 'failed':
        raise HTTPException(status_code=500, detail="Ingestion pipeline failed")
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    logger.info(f"Query request received: {request.query[:50]}...")
    result = run_query_pipeline(request.query)
    if not result.get('answer'):
        raise HTTPException(status_code=500, detail="Query pipeline failed")
    return QueryResponse(**result)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    logger.info(f"Evaluate request received for {len(request.eval_dataset)} samples")
    eval_dataset = [item.model_dump() for item in request.eval_dataset]
    result = evaluate_pipeline(eval_dataset)
    if result['status'] == 'failed':
        raise HTTPException(status_code=500, detail="Evaluation pipeline failed")
    return EvaluateResponse(**result)
