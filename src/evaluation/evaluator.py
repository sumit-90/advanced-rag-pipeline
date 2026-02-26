from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from src.pipeline import run_query_pipeline
from src.retrieval.retriever import retrieve_documents
import json
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()

def evaluate_pipeline(eval_dataset:list[dict])->dict:
    try:
        logger.info(f"Starting evaluation for {len(eval_dataset)} samples")

        if not eval_dataset:
            logger.warning("No evaluation dataset provided")
            return {"status": "failed", "results": {}}
        
        samples = []
        for item in eval_dataset:
            answer = run_query_pipeline(item['question'])
            contexts = [doc['content'] for doc in retrieve_documents(item['question'])]
            samples.append(SingleTurnSample(
                user_input=item['question'],
                response=answer['answer'],
                retrieved_contexts=contexts,
                reference=item['ground_truth']
            ))
        
        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]
        dataset = EvaluationDataset(samples=samples)
        results = evaluate(dataset, metrics=metrics)
        logger.info(f"Evaluation scores: {json.dumps(str(results), indent=2)}")

        results_summary = {
            "scores": results,
            "num_evaluated_samples": len(samples)
        }
        
        return {"status": "success", "results": results_summary}
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        return {"status": "failed", "results": {}}