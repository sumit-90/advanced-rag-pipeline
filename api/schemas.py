from pydantic import BaseModel
from typing import Any

# Request models
class IngestRequest(BaseModel):
    directory: str

class QueryRequest(BaseModel):
    query: str

class EvalSample(BaseModel):
    question: str
    ground_truth: str

class EvaluateRequest(BaseModel):
    eval_dataset: list[EvalSample]

# Response models
class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    collection: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str

class EvaluateResponse(BaseModel):
    status: str
    results: Any
