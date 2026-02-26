from src.guardrails.guardrails import validate_query, validate_response


def test_validate_query_empty():
    is_valid, _ = validate_query("")
    assert not is_valid

def test_validate_query_too_short():
    is_valid, _ = validate_query("short")
    assert not is_valid

def test_validate_query_too_long():
    is_valid, _ = validate_query("a" * 513)
    assert not is_valid

def test_validate_query_numeric_only():
    is_valid, _ = validate_query("123456789")
    assert not is_valid

def test_validate_query_valid():
    is_valid, _ = validate_query("What is the capital of France?")
    assert is_valid


def test_validate_response_empty_answer():
    is_valid, _ = validate_response({"answer": "", "sources": ["doc.pdf"], "model": "gpt-4.1-mini"})
    assert not is_valid

def test_validate_response_no_sources():
    is_valid, _ = validate_response({"answer": "Paris is the capital.", "sources": [], "model": "gpt-4.1-mini"})
    assert not is_valid

def test_validate_response_i_dont_know():
    is_valid, _ = validate_response({"answer": "I don't know based on the provided documents.", "sources": ["doc.pdf"], "model": "gpt-4.1-mini"})
    assert not is_valid

def test_validate_response_valid():
    is_valid, _ = validate_response({"answer": "Paris is the capital of France.", "sources": ["doc.pdf"], "model": "gpt-4.1-mini"})
    assert is_valid
