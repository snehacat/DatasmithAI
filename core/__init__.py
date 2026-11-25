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
    EvaluationRequirements,
    ApplicationPlanningPolicy,
    UsageIntent,
    EvaluationDimension,
    Criticality,
    DiscoveryMethod,
    RetrievalMethod,
    UsageRight,
    EvidenceStatus,
    EvidenceType,
    EvidenceCaptureMethod,
    EvidenceAuthority,
    SourceEvaluationStatus,
    AccessClass,
    AuthenticationRequirement,
    QuotaStatus,
    LicensePermissionStatus,
    LicenseConditionsStatus,
    ConnectorAssessmentStatus,
    SourceEvidence,
    SourceFinding,
    QuotaAssessment,
    LicensePermissionAssessment,
    ConnectorAssessment,
    SourceAssessment,
    CandidateSource,
    CandidateSelectionStatus,
    derive_evaluation_requirements,
    validate_spec,
    create_initial_spec
)

from .llm_wrapper import (
    LLMWrapper,
    LLMConfig,
    LLMResponse,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMResponseError,
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
    'EvaluationRequirements',
    'ApplicationPlanningPolicy',
    'UsageIntent',
    'EvaluationDimension',
    'Criticality',
    'DiscoveryMethod',
    'RetrievalMethod',
    'UsageRight',
    'EvidenceStatus',
    'EvidenceType',
    'EvidenceCaptureMethod',
    'EvidenceAuthority',
    'SourceEvaluationStatus',
    'AccessClass',
    'AuthenticationRequirement',
    'QuotaStatus',
    'LicensePermissionStatus',
    'LicenseConditionsStatus',
    'ConnectorAssessmentStatus',
    'SourceEvidence',
    'SourceFinding',
    'QuotaAssessment',
    'LicensePermissionAssessment',
    'ConnectorAssessment',
    'SourceAssessment',
    'CandidateSource',
    'CandidateSelectionStatus',
    'derive_evaluation_requirements',
    'validate_spec',
    'create_initial_spec',
    
    # LLM wrapper
    'LLMWrapper',
    'LLMConfig',
    'LLMResponse',
    'LLMError',
    'LLMConnectionError',
    'LLMTimeoutError',
    'LLMResponseError',
    'get_llm',
    'reset_llm',
    
    # Orchestrator
    'Orchestrator',
    'AgentDefinition',
    'create_orchestrator'
]
