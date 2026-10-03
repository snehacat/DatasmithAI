"""
Core foundations for DataSmith AI.

Phase 0: Shared infrastructure used by all agents.
"""

from .dataset_spec import (
    DatasetSpec,
    RequirementSpec,
    PlanSpec,
    RetrievalSpec,
    SourceRecord,
    DataRecord,
    AgentExecution,
    DataModality,
    PipelineStatus,
    AgentStatus,
    validate_spec,
    create_initial_spec
)

from .llm_wrapper import (
    LLMWrapper,
    LLMConfig,
    LLMResponse,
    get_llm,
    reset_llm
)

from .orchestrator import (
    Orchestrator,
    AgentDefinition,
    create_orchestrator
)

__all__ = [
    # Data contract
    'DatasetSpec',
    'RequirementSpec',
    'PlanSpec',
    'RetrievalSpec',
    'SourceRecord',
    'DataRecord',
    'AgentExecution',
    'DataModality',
    'PipelineStatus',
    'AgentStatus',
    'validate_spec',
    'create_initial_spec',
    
    # LLM wrapper
    'LLMWrapper',
    'LLMConfig',
    'LLMResponse',
    'get_llm',
    'reset_llm',
    
    # Orchestrator
    'Orchestrator',
    'AgentDefinition',
    'create_orchestrator'
]
