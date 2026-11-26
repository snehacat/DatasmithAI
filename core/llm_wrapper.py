"""
DataSmith AI - LLM Wrapper

Unified interface for Ollama/Qwen calls.
All agents use this instead of calling ollama directly.

Features:
- JSON-only output
- Retries on failure
- Timeouts (per-attempt and total)
- Logging
- Output validation
"""

import json
import threading
import time
import logging
import os
from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field, ValidationError
import httpx
import ollama
from ollama import Client as OllamaClient

_REAL_OLLAMA_CLIENT = OllamaClient


# Setup logging
logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for expected failures while requesting an LLM response."""


class LLMConnectionError(LLMError):
    """The configured Ollama service could not be reached."""


class LLMTimeoutError(LLMError, TimeoutError):
    """An LLM request or total call exceeded its timeout."""


class LLMResponseError(LLMError):
    """The LLM service returned an unusable or invalid response."""


class LLMConfig(BaseModel):
    """Configuration for LLM calls."""
    model_name: str = "qwen2.5:3b"
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 800   # analyzer JSON is ~150 tokens; 800 is a safe cap
    num_ctx: int = 4096      # prompt is ~600 tokens; a smaller context is faster on CPU
    total_timeout_seconds: float = Field(
        120.0,
        ge=0.0,
        description=(
            "Total retry-budget threshold checked between attempts and used to bound "
            "each attempt's transport timeout (not a hard mid-stream wall-clock interrupt)."
        ),
    )
    per_attempt_timeout_seconds: float = Field(
        60.0,
        gt=0.0,
        description="Per-attempt transport inactivity timeout (connect/read/write/pool) in seconds.",
    )
    max_retries: int = Field(1, ge=0)  # Maximum 1 retry (2 total attempts)
    keep_alive: str = "30m"  # keep the model loaded between calls (avoids 15-25s reloads)

class LLMResponse(BaseModel):
    """Successful LLM response only; failures are raised instead of returned."""
    success: bool
    content: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    model_used: str
    duration_seconds: float
    retries_used: int = 0


class LLMWrapper:
    """
    Wrapper for local LLM calls via Ollama.
    
    Usage:
        llm = LLMWrapper()
        response = llm.call(prompt, expected_schema=MyPydanticModel)
        if response.success:
            data = response.content
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM wrapper.
        
        Args:
            config: LLM configuration (uses defaults if None)
        """
        self.config = config or LLMConfig()
        self.call_count = 0
        self._call_count_lock = threading.Lock()
        self._host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = self._build_client(self.config.per_attempt_timeout_seconds)

        logger.info(
            f"LLMWrapper initialized with model: {self.config.model_name}, "
            f"timeout: {self.config.per_attempt_timeout_seconds}s, "
            f"num_ctx: {self.config.num_ctx}, num_predict: {self.config.num_predict}"
        )

    def _build_client(self, timeout_seconds: float) -> Any:
        """Create an OllamaClient bound to an explicit finite transport timeout."""
        bounded_timeout = max(0.001, float(timeout_seconds))
        return OllamaClient(
            host=self._host,
            timeout=httpx.Timeout(bounded_timeout),
        )

    def _get_attempt_client(self, attempt_timeout: float) -> Tuple[Any, bool]:
        """Return (client, should_close) enforcing attempt_timeout without mutating shared client."""
        if self.client is None:
            return self._build_client(attempt_timeout), True

        is_managed_client = isinstance(self.client, _REAL_OLLAMA_CLIENT) or (
            isinstance(OllamaClient, type) and isinstance(self.client, OllamaClient)
        )
        if not is_managed_client:
            # Test fakes/spies injected onto llm.client are used directly.
            return self.client, False

        if attempt_timeout < self.config.per_attempt_timeout_seconds:
            # Use a request-scoped client so shared self.client timeout state is never mutated.
            return self._build_client(attempt_timeout), True

        return self.client, False

    @staticmethod
    def _close_client(client: Any) -> None:
        """Close a request-scoped OllamaClient's underlying httpx client safely."""
        for target in (getattr(client, "_client", None), client):
            close_fn = getattr(target, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                    return
                except Exception:
                    pass
    
    def call(
        self,
        prompt: str,
        expected_schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        force_json: bool = True
    ) -> LLMResponse:
        """
        Call LLM with retries and validation.

        Timeout semantics:
        - Before each attempt, remaining ``total_timeout_seconds`` is checked and used
          to cap ``attempt_timeout = min(per_attempt_timeout_seconds, remaining_total)``.
        - Each attempt uses a bounded ``httpx.Timeout(attempt_timeout)`` transport
          inactivity timeout (connect/read/write/pool). A synchronous response that
          continuously makes progress without triggering transport inactivity is not
          force-cancelled mid-request, though any subsequent retry checks the total budget.
        
        Args:
            prompt: The user prompt
            expected_schema: Optional Pydantic model to validate against
            system_prompt: Optional system prompt
            force_json: Force JSON output mode
            
        Returns:
            LLMResponse with parsed content. On success, success is always True;
            failures are raised, never returned.

        Raises:
            LLMConnectionError: Ollama is unreachable.
            LLMTimeoutError: a transport timeout or inter-attempt total budget was exceeded.
            LLMResponseError: all response parsing or validation attempts failed.
            LLMError: Ollama returned an unsuccessful provider response.
        """
        with self._call_count_lock:
            self.call_count += 1
            call_num = self.call_count
        overall_start_time = time.monotonic()
        
        logger.info(f"LLM call #{call_num} starting...")
        logger.debug(f"Prompt: {prompt[:200]}...")
        
        last_error = None
        validation_error = None
        validation_cause = None
        
        for attempt in range(self.config.max_retries + 1):  # max_retries + initial attempt
            # Check total timeout
            elapsed = time.monotonic() - overall_start_time
            if elapsed >= self.config.total_timeout_seconds:
                duration = time.monotonic() - overall_start_time
                error_msg = (
                    f"LLM call exceeded total timeout of {self.config.total_timeout_seconds}s "
                    f"after {attempt} attempt(s). Total time: {duration:.2f}s"
                )
                logger.error(error_msg)
                raise LLMTimeoutError(error_msg)
            
            # Calculate remaining time for this attempt
            remaining_total = self.config.total_timeout_seconds - elapsed
            attempt_timeout = min(self.config.per_attempt_timeout_seconds, remaining_total)
            
            if attempt_timeout <= 0:
                duration = time.monotonic() - overall_start_time
                error_msg = (
                    f"LLM call exceeded total timeout of {self.config.total_timeout_seconds}s "
                    f"before attempt {attempt + 1}. Total time: {duration:.2f}s"
                )
                logger.error(error_msg)
                raise LLMTimeoutError(error_msg)
            
            attempt_start = time.monotonic()
            
            try:
                # Build messages
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                
                # On retry, include validation error in prompt
                if attempt > 0 and validation_error:
                    retry_prompt = (
                        f"Previous attempt failed validation: {validation_error}\n\n"
                        f"Please fix and respond with valid JSON: {prompt}"
                    )
                    messages.append({'role': 'user', 'content': retry_prompt})
                    logger.debug(f"Retry with error feedback: {validation_error[:100]}")
                else:
                    messages.append({'role': 'user', 'content': prompt})
                
                # Call Ollama
                logger.debug(f"Attempt {attempt + 1}/{self.config.max_retries + 1}")
                
                try:
                    call_start = time.monotonic()
                    attempt_client, should_close = self._get_attempt_client(attempt_timeout)
                    try:
                        response = attempt_client.chat(
                            model=self.config.model_name,
                            messages=messages,
                            format='json' if force_json else None,
                            options={
                                'temperature': self.config.temperature,
                                'top_p': self.config.top_p,
                                'num_predict': self.config.num_predict,
                                'num_ctx': self.config.num_ctx
                            },
                            keep_alive=self.config.keep_alive
                        )
                    finally:
                        if should_close:
                            self._close_client(attempt_client)

                    call_duration = time.monotonic() - call_start

                    # Check if this single call exceeded the per-attempt timeout
                    if call_duration > attempt_timeout:
                        logger.warning(
                            f"Call took {call_duration:.2f}s, exceeded attempt timeout of {attempt_timeout:.2f}s"
                        )

                except (httpx.TimeoutException, TimeoutError) as e:
                    error_msg = (
                        f"Ollama request timed out after {time.monotonic() - overall_start_time:.2f}s. "
                        f"Error: {e}"
                    )
                    logger.error(error_msg)
                    raise LLMTimeoutError(error_msg) from e
                except (ConnectionError, httpx.TransportError) as e:
                    duration = time.monotonic() - overall_start_time
                    error_msg = (
                        f"Cannot connect to Ollama service after {duration:.2f}s. "
                        f"Please ensure Ollama is running (ollama serve). "
                        f"Error: {e}"
                    )
                    logger.error(error_msg)
                    raise LLMConnectionError(error_msg) from e
                except (json.JSONDecodeError, httpx.DecodingError, ValidationError) as e:
                    raise LLMResponseError(
                        f"Ollama returned an invalid response: {e}"
                    ) from e
                except ollama.ResponseError as e:
                    message = str(e).lower()
                    if "model" in message and ("not found" in message or "does not exist" in message):
                        error_msg = f"Run: ollama pull {self.config.model_name}"
                        logger.error(error_msg)
                        raise LLMError(error_msg) from e
                    raise LLMResponseError(f"Ollama returned an error response: {e}") from e
                
                # Extract content
                try:
                    raw_text = response["message"]["content"]
                except (KeyError, TypeError) as e:
                    validation_error = f"invalid response structure: {e}"
                    last_error = validation_error
                    validation_cause = e
                    logger.warning(validation_error)
                    if attempt >= self.config.max_retries:
                        break
                    continue
                if not isinstance(raw_text, str):
                    validation_error = "response content was not a string"
                    last_error = validation_error
                    validation_cause = None
                    logger.warning(validation_error)
                    if attempt >= self.config.max_retries:
                        break
                    continue
                self._log_ollama_stats(response)
                attempt_duration = time.monotonic() - attempt_start
                logger.debug(f"Attempt {attempt + 1} completed in {attempt_duration:.2f}s")
                logger.debug(f"Raw response: {raw_text[:200]}...")
                
                # Parse JSON
                try:
                    content = json.loads(raw_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed: {e}")
                    last_error = f"JSON parse error: {e}"
                    validation_error = f"response was not valid JSON ({e})"
                    validation_cause = e
                    if attempt >= self.config.max_retries:
                        break
                    continue
                                    # The model must return a JSON object ({...}), not a list or a bare value
                if not isinstance(content, dict):
                    validation_error = (
                        f"expected a JSON object, got {type(content).__name__}"
                    )
                    logger.warning(f"Unusable JSON shape: {validation_error}")
                    last_error = f"JSON shape error: {validation_error}"
                    validation_cause = None
                    if attempt >= self.config.max_retries:
                        break
                    continue
                
                # Validate against schema if provided
                if expected_schema:
                    try:
                        validated = expected_schema(**content)
                        # pydantic v2: model_dump(); v1 fallback: dict()
                        content = validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
                        logger.debug("Schema validation passed")
                    except ValidationError as e:
                        validation_error = str(e)
                        logger.warning(f"Schema validation failed: {validation_error[:200]}")
                        last_error = f"Schema validation error: {validation_error}"
                        validation_cause = e
                        if attempt >= self.config.max_retries:
                            break
                        continue
                
                # Success!
                duration = time.monotonic() - overall_start_time
                logger.info(f"✅ LLM call succeeded in {duration:.2f}s (attempt {attempt + 1})")
                
                return LLMResponse(
                    success=True,
                    content=content,
                    raw_text=raw_text,
                    model_used=self.config.model_name,
                    duration_seconds=duration,
                    retries_used=attempt
                )
                
            except LLMError:
                raise
        
        # All attempts exhausted
        duration = time.monotonic() - overall_start_time
        error_msg = (
            f"LLM call failed after {self.config.max_retries + 1} attempt(s) "
            f"in {duration:.2f}s. Last error: {last_error}"
        )
        logger.error(error_msg)
        if validation_cause is not None:
            raise LLMResponseError(error_msg) from validation_cause
        raise LLMResponseError(error_msg)
    @staticmethod
    def _log_ollama_stats(response):
        """Log Ollama's own numbers: tokens in/out and where the time went."""
        try:
            prompt_tokens = response.get('prompt_eval_count')
            output_tokens = response.get('eval_count')
            if prompt_tokens is None or output_tokens is None:
                return

            def seconds(ns):
                return (ns or 0) / 1e9

            logger.info(
                f"Ollama stats: prompt_tokens={prompt_tokens} output_tokens={output_tokens} "
                f"load={seconds(response.get('load_duration')):.1f}s "
                f"prompt_eval={seconds(response.get('prompt_eval_duration')):.1f}s "
                f"generate={seconds(response.get('eval_duration')):.1f}s"
            )
        except Exception:
            pass   # statistics must never break a real call
    
    def get_call_count(self) -> int:
        """Get total number of LLM calls made."""
        with self._call_count_lock:
            return self.call_count
    
    def reset_call_count(self):
        """Reset call counter."""
        with self._call_count_lock:
            self.call_count = 0


# Global singleton instance
_global_llm: Optional[LLMWrapper] = None


def get_llm(config: Optional[LLMConfig] = None) -> LLMWrapper:
    """
    Get or create global LLM wrapper instance.
    
    Args:
        config: Configuration (only used on first call)
        
    Returns:
        LLMWrapper instance
    """
    global _global_llm
    if _global_llm is None:
        _global_llm = LLMWrapper(config)
    return _global_llm


def reset_llm():
    """Reset global LLM wrapper (useful for testing)."""
    global _global_llm
    _global_llm = None