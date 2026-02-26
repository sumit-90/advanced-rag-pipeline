from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config_loader import load_config
from src.logger import get_logger
logger = get_logger(__name__)
config = load_config()


def generate_response(query:str, documents:list[dict]) -> dict:
    if not query:
        logger.warning("No query provided")
        return {"answer": "", "sources": [], "model": config['llm']['model_name']}

    if not documents:
        logger.warning("No documents provided")
        return {
            "answer": "I don't have enough context to answer this question.",
            "sources": [],
            "model": config['llm']['model_name']
        }

    # For simplicity, we will just concatenate the retrieved documents and use them as context for the response.
    # In a real implementation, you would likely want to use a more sophisticated approach to generate the response.
    context = ""
    for i, doc in enumerate(documents):
        context += f"[{i+1}] {doc['content']}\n\n"

    system_prompt = """You are a helpful assistant. 
    Answer the question using ONLY the provided context.
    If the answer is not in the context, say 'I don't know based on the provided documents.'"""

    human_prompt = f"""Context:
    {context}

    Question: {query}"""

    llm = ChatOpenAI(
        model=config['llm']['model_name'],
        openai_api_key=config['credentials']['openai_api_key'],
        temperature=config['llm']['temperature']
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    logger.info(f"Generating response for query: {query[:50]}...")
    response = llm.invoke(messages)

    logger.info(f"Response generated — length: {len(response.content)} characters")
    
    # deduplicate sources
    sources = list(set([doc['metadata']['source'] for doc in documents]))

    return {
        "answer": response.content,
        "sources": sources,
        "model": config['llm']['model_name']
    }
