"""
Tests for LLM Wrapper
"""

import json
import logging

import httpx
import ollama
import pytest
from pydantic import BaseModel
from pydantic import ValidationError
from core.llm_wrapper import (
    LLMConnectionError,
    LLMConfig,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    LLMWrapper,
    get_llm,
    reset_llm,
)


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
    assert llm.call_count == 1


@pytest.mark.skip(reason="Requires Ollama running with qwen2.5:3b")
def test_schema_validation():
    """Test schema validation catches invalid responses."""
    llm = LLMWrapper()

    # Missing 'name' field: fails validation on every attempt, so the wrapper raises
    with pytest.raises(LLMResponseError, match="validation"):
        llm.call(
            prompt='Return JSON: {"value": 42, "success": true}',
            expected_schema=SampleSchema
        )
    
   


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
       # Invalid JSON fails parsing on every attempt, so the wrapper raises
    with pytest.raises(LLMResponseError, match="JSON"):
        llm.call(
            prompt='Return invalid JSON: {name: test, value: invalid}',
            force_json=False  # Don't force JSON format
        )

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
class FakeClient:
    """Stands in for the Ollama client; returns the queued replies in order."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        return {"message": {"content": self.replies.pop(0)}}


def make_llm(replies):
    llm = LLMWrapper(LLMConfig(max_retries=1))
    llm.client = FakeClient(replies)
    return llm


def test_non_object_json_is_rejected_with_clear_error():
    llm = make_llm(['[1, 2, 3]', '[1, 2, 3]'])
    with pytest.raises(LLMResponseError, match="expected a JSON object, got list"):
        llm.call("give me json")
    assert len(llm.client.prompts) == 2   # it retried once


def test_retry_after_non_object_json_can_succeed():
    llm = make_llm(['[1, 2, 3]', '{"ok": true}'])
    response = llm.call("give me json")
    assert response.success
    assert response.content == {"ok": True}
    assert response.retries_used == 1
    assert "expected a JSON object" in llm.client.prompts[1]   # feedback sent to the model


def test_invalid_json_retry_includes_feedback():
    llm = make_llm(['not json at all', '{"ok": true}'])
    response = llm.call("give me json")
    assert response.content == {"ok": True}
    assert "not valid JSON" in llm.client.prompts[1]

def test_keep_alive_is_sent_to_ollama():
    seen = {}

    class Spy:
        def chat(self, **kwargs):
            seen.update(kwargs)
            return {"message": {"content": '{"ok": true}'}}

    llm = LLMWrapper(LLMConfig(keep_alive="30m"))
    llm.client = Spy()
    llm.call("hello")
    assert seen["keep_alive"] == "30m"

def test_ollama_stats_are_logged(caplog):
    class Spy:
        def chat(self, **kwargs):
            return {
                "message": {"content": '{"ok": true}'},
                "prompt_eval_count": 600, "eval_count": 150,
                "load_duration": 0,
                "prompt_eval_duration": 8_000_000_000,
                "eval_duration": 30_000_000_000,
            }

    llm = LLMWrapper()
    llm.client = Spy()
    with caplog.at_level(logging.INFO, logger="core.llm_wrapper"):
        response = llm.call("hello")

    assert response.success
    assert "output_tokens=150" in caplog.text
    assert "generate=30.0s" in caplog.text


def test_ollama_host_comes_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example:11434")
    seen = {}

    class SpyClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("core.llm_wrapper.OllamaClient", SpyClient)
    llm = LLMWrapper()
    assert seen["host"] == "http://ollama.example:11434"


def test_httpx_connect_error_is_reported_as_connection_failure():
    class Spy:
        def chat(self, **kwargs):
            raise httpx.ConnectError("connection refused")

    llm = LLMWrapper()
    llm.client = Spy()
    with pytest.raises(LLMConnectionError, match="Cannot connect to Ollama") as caught:
        llm.call("hello")
    assert isinstance(caught.value.__cause__, httpx.ConnectError)


def test_httpx_request_timeout_uses_timeout_error_and_preserves_cause():
    timeout = httpx.ReadTimeout("request timed out")

    class Spy:
        def chat(self, **kwargs):
            raise timeout

    llm = LLMWrapper()
    llm.client = Spy()
    with pytest.raises(LLMTimeoutError) as caught:
        llm.call("hello")

    assert isinstance(caught.value, TimeoutError)
    assert caught.value.__cause__ is timeout


def test_malformed_json_uses_response_error_and_preserves_cause():
    llm = make_llm(["not json", "still not json"])
    with pytest.raises(LLMResponseError) as caught:
        llm.call("give me json")

    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "error",
    [
        json.JSONDecodeError("invalid response JSON", "{", 0),
        httpx.DecodingError("invalid response encoding"),
    ],
)
def test_client_response_decoding_errors_use_response_error(error):
    class Spy:
        def chat(self, **kwargs):
            raise error

    llm = LLMWrapper()
    llm.client = Spy()
    with pytest.raises(LLMResponseError) as caught:
        llm.call("hello")

    assert caught.value.__cause__ is error


def test_client_response_validation_error_uses_response_error():
    with pytest.raises(ValidationError) as validation:
        SampleSchema.model_validate({})
    error = validation.value

    class Spy:
        def chat(self, **kwargs):
            raise error

    llm = LLMWrapper()
    llm.client = Spy()
    with pytest.raises(LLMResponseError) as caught:
        llm.call("hello")

    assert caught.value.__cause__ is error


def test_total_timeout_uses_custom_timeout_error():
    llm = LLMWrapper(LLMConfig(total_timeout_seconds=0))

    with pytest.raises(LLMTimeoutError) as caught:
        llm.call("hello")

    assert caught.value.__cause__ is None


def test_schema_validation_exhaustion_uses_response_error_and_preserves_cause():
    llm = make_llm(['{"value": 42, "success": true}'] * 2)

    with pytest.raises(LLMResponseError) as caught:
        llm.call("give me json", expected_schema=SampleSchema)

    assert isinstance(caught.value.__cause__, ValidationError)


def test_invalid_response_structure_uses_response_error_and_preserves_cause():
    class Spy:
        def chat(self, **kwargs):
            return {"message": {}}

    llm = LLMWrapper()
    llm.client = Spy()
    with pytest.raises(LLMResponseError) as caught:
        llm.call("hello")

    assert isinstance(caught.value.__cause__, KeyError)


@pytest.mark.parametrize("error", [
    TypeError("unexpected type bug"),
    KeyError("unexpected key bug"),
    RuntimeError("unrelated runtime bug"),
])
def test_unexpected_client_errors_are_not_reclassified(error):
    class Spy:
        def chat(self, **kwargs):
            raise error

    llm = LLMWrapper()
    llm.client = Spy()
    with pytest.raises(type(error)) as caught:
        llm.call("hello")

    assert caught.value is error


def test_model_not_found_has_pull_command():
    class Spy:
        def chat(self, **kwargs):
            raise ollama.ResponseError("model 'missing-model' not found", status_code=404)

    llm = LLMWrapper(LLMConfig(model_name="missing-model"))
    llm.client = Spy()
    with pytest.raises(LLMError, match=r"Run: ollama pull missing-model") as caught:
        llm.call("hello")
    assert isinstance(caught.value.__cause__, ollama.ResponseError)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
