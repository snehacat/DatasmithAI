"""
DataSmith AI - LLM Wrapper

Unified interface for Ollama/Qwen calls.
All agents use this instead of calling ollama directly.

Features:
- JSON-only output
- Retries on failure
- Timeouts
- Logging
- Output validation
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
import ollama


# Setup logging
logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM calls."""
    model_name: str = "qwen2.5:3b"
    temperature: float = 0.2
    top_p: float = 0.9
    num_predict: int = 2000
    num_ctx: int = 8192
    total_timeout_seconds: int = 60  # Total time limit for all attempts
    max_retries: int = 1  # Maximum 1 retry (2 total attempts)


class LLMResponse(BaseModel):
    """Standard LLM response."""
    success: bool
    content: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None
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
        
        logger.info(f"LLMWrapper initialized with model: {self.config.model_name}")
    
    def call(
        self,
        prompt: str,
        expected_schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        force_json: bool = True
    ) -> LLMResponse:
        """
        Call LLM with retries and validation.
        
        Args:
            prompt: The user prompt
            expected_schema: Optional Pydantic model to validate against
            system_prompt: Optional system prompt
            force_json: Force JSON output mode
            
        Returns:
            LLMResponse with success/failure and parsed content
        """
        self.call_count += 1
        overall_start_time = time.time()
        
        logger.info(f"LLM call #{self.call_count} starting...")
        logger.debug(f"Prompt: {prompt[:200]}...")
        
        last_error = None
        validation_error = None
        
        for attempt in range(self.config.max_retries + 1):  # max_retries + initial attempt
            # Check total timeout
            elapsed = time.time() - overall_start_time
            if elapsed >= self.config.total_timeout_seconds:
                duration = time.time() - overall_start_time
                error_msg = (
                    f"LLM call exceeded total timeout of {self.config.total_timeout_seconds}s "
                    f"after {attempt} attempt(s). Total time: {duration:.2f}s"
                )
                logger.error(error_msg)
                raise TimeoutError(error_msg)
            
            attempt_start = time.time()
            
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
                    response = ollama.chat(
                        model=self.config.model_name,
                        messages=messages,
                        format='json' if force_json else None,
                        options={
                            'temperature': self.config.temperature,
                            'top_p': self.config.top_p,
                            'num_predict': self.config.num_predict,
                            'num_ctx': self.config.num_ctx
                        }
                    )
                except ConnectionError as e:
                    # Connection error - fail immediately
                    duration = time.time() - overall_start_time
                    error_msg = (
                        f"Cannot connect to Ollama service after {duration:.2f}s. "
                        f"Please ensure Ollama is running (ollama serve). "
                        f"Error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                except Exception as e:
                    # Check if it's a connection-related error
                    if 'connection' in str(e).lower() or 'refused' in str(e).lower():
                        duration = time.time() - overall_start_time
                        error_msg = (
                            f"Cannot connect to Ollama service after {duration:.2f}s. "
                            f"Please ensure Ollama is running (ollama serve). "
                            f"Error: {e}"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg) from e
                    raise
                
                # Extract content
                raw_text = response['message']['content']
                attempt_duration = time.time() - attempt_start
                logger.debug(f"Attempt {attempt + 1} completed in {attempt_duration:.2f}s")
                logger.debug(f"Raw response: {raw_text[:200]}...")
                
                # Parse JSON
                try:
                    content = json.loads(raw_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed: {e}")
                    last_error = f"JSON parse error: {e}"
                    if attempt >= self.config.max_retries:
                        break
                    continue
                
                # Validate against schema if provided
                if expected_schema:
                    try:
                        validated = expected_schema(**content)
                        content = validated.dict()
                        logger.debug("Schema validation passed")
                    except Exception as e:
                        validation_error = str(e)
                        logger.warning(f"Schema validation failed: {validation_error[:200]}")
                        last_error = f"Schema validation error: {validation_error}"
                        if attempt >= self.config.max_retries:
                            break
                        continue
                
                # Success!
                duration = time.time() - overall_start_time
                logger.info(f"✅ LLM call succeeded in {duration:.2f}s (attempt {attempt + 1})")
                
                return LLMResponse(
                    success=True,
                    content=content,
                    raw_text=raw_text,
                    model_used=self.config.model_name,
                    duration_seconds=duration,
                    retries_used=attempt
                )
                
            except (RuntimeError, TimeoutError):
                # Re-raise connection errors and timeouts immediately
                raise
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error[:200]}")
                
                if attempt >= self.config.max_retries:
                    break
        
        # All attempts exhausted
        duration = time.time() - overall_start_time
        error_msg = (
            f"LLM call failed after {self.config.max_retries + 1} attempt(s) "
            f"in {duration:.2f}s. Last error: {last_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def get_call_count(self) -> int:
        """Get total number of LLM calls made."""
        return self.call_count
    
    def reset_call_count(self):
        """Reset call counter."""
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
