"""
Tests for LLM Wrapper
"""

import pytest
from pydantic import BaseModel
from core.llm_wrapper import LLMWrapper, LLMConfig, get_llm, reset_llm


class SampleSchema(BaseModel):
    """Sample schema for testing."""
    name: str
    value: int
    success: bool


def test_llm_wrapper_initialization():
    """Test LLM wrapper initialization."""
    config = LLMConfig(model_name="qwen2.5:3b")
    llm = LLMWrapper(config)
    
    assert llm.config.model_name == "qwen2.5:3b"
    assert llm.call_count == 0


def test_global_llm_singleton():
    """Test global LLM singleton."""
    reset_llm()
    
    llm1 = get_llm()
    llm2 = get_llm()
    
    assert llm1 is llm2  # Same instance


@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
def test_successful_llm_call():
    """Test successful LLM call with JSON output."""
    llm = LLMWrapper()
    
    response = llm.call(
        prompt='Return JSON: {"name": "test", "value": 42, "success": true}',
        expected_schema=SampleSchema
    )
    
    assert response.success
    assert response.content is not None
    assert response.content['name'] == 'test'
    assert response.content['value'] == 42
    assert response.error is None
    assert llm.call_count == 1


@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
def test_schema_validation():
    """Test schema validation catches invalid responses."""
    llm = LLMWrapper()
    
    # This should fail schema validation (missing 'name' field)
    response = llm.call(
        prompt='Return JSON: {"value": 42, "success": true}',
        expected_schema=SampleSchema
    )
    
    # Should fail after retries
    assert not response.success
    assert response.error is not None
    assert "validation" in response.error.lower() or "missing" in response.error.lower()


def test_call_count_tracking():
    """Test that call count is tracked."""
    llm = LLMWrapper()
    
    assert llm.get_call_count() == 0
    
    # Even failed calls should increment
    # (This won't actually call LLM without Ollama, but tests the counter)
    llm.call_count = 5
    assert llm.get_call_count() == 5
    
    llm.reset_call_count()
    assert llm.get_call_count() == 0


def test_retry_configuration():
    """Test retry configuration."""
    config = LLMConfig(
        max_retries=1,  # Changed from 5 to match new max
        total_timeout_seconds=30  # Changed from timeout_seconds
    )
    llm = LLMWrapper(config)
    
    assert llm.config.max_retries == 1
    assert llm.config.total_timeout_seconds == 30


@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
def test_invalid_json_handling():
    """Test handling of invalid JSON responses."""
    llm = LLMWrapper()
    
    # Force non-JSON mode and ask for invalid JSON
    response = llm.call(
        prompt='Return invalid JSON: {name: test, value: invalid}',
        force_json=False  # Don't force JSON format
    )
    
    # Should fail JSON parsing after retries
    assert not response.success or response.content is None


@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
def test_llm_with_system_prompt():
    """Test LLM call with system prompt."""
    llm = LLMWrapper()
    
    response = llm.call(
        system_prompt="You are a helpful assistant. Always respond with valid JSON.",
        prompt='Return: {"name": "assistant", "value": 1, "success": true}',
        expected_schema=SampleSchema
    )
    
    assert response.success
    assert response.content is not None


def test_response_model_structure():
    """Test that response model has expected fields."""
    from core.llm_wrapper import LLMResponse
    
    # Create a mock response
    response = LLMResponse(
        success=True,
        content={"test": "data"},
        model_used="qwen2.5:3b",
        duration_seconds=2.5,
        retries_used=0
    )
    
    assert response.success is True
    assert response.content == {"test": "data"}
    assert response.model_used == "qwen2.5:3b"
    assert response.duration_seconds == 2.5
    assert response.retries_used == 0
    assert response.error is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
