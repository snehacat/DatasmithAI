"""
DataSmith AI - Pipeline Orchestrator

Runs agents in order, passes DatasetSpec through pipeline,
saves after each step, supports resume from any point.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime

from .dataset_spec import (
    DatasetSpec,
    AgentExecution,
    AgentStatus,
    PipelineStatus,
    validate_spec
)
from .run_manager import get_run_manager


# Setup logging
logger = logging.getLogger(__name__)


class AgentDefinition:
    """Definition of an agent in the pipeline."""
    
    def __init__(
        self,
        name: str,
        run_func: Callable[[DatasetSpec], DatasetSpec],
        required_inputs: List[str],
        outputs: List[str],
        skip_if_unsupported: bool = True,
        stats_func: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        """
        Define an agent.
        
        Args:
            name: Agent name (e.g., "RequirementAnalyzer")
            run_func: Function that takes spec and returns updated spec
            required_inputs: Which spec sections must exist before this runs
            outputs: Which spec sections this agent writes
            skip_if_unsupported: Skip if requirement is unsupported
            stats_func: Optional function returning the agent's run statistics
                        {'llm_calls', 'llm_seconds', 'user_wait_seconds'}.
                        If not given, the orchestrator looks for a `stats` dict on
                        the object that owns run_func (e.g. RequirementAnalyzer.stats).
        """
        self.name = name
        self.run_func = run_func
        self.required_inputs = required_inputs
        self.outputs = outputs
        self.skip_if_unsupported = skip_if_unsupported
        self.stats_func = stats_func


class Orchestrator:
    """
    Pipeline orchestrator.
    
    Responsibilities:
    - Run agents in order
    - Pass DatasetSpec between agents
    - Save spec after each agent (for resume)
    - Handle errors and skip conditions
    - Mark rejected requests as REJECTED (later agents are SKIPPED, status is kept)
    - Record LLM time and user-wait time separately for each agent
    - Comprehensive logging
    
    Usage:
        orchestrator = Orchestrator(save_dir="./runs")
        orchestrator.register_agent(agent_def)
        spec = orchestrator.run(dataset_name, description)
    """
    
    def __init__(
        self,
        save_dir: str = "./runs",  # Deprecated, use run_manager instead
        auto_save: bool = True,
        validate_after_each: bool = True
    ):
        """
        Initialize orchestrator.
        
        Args:
            save_dir: Deprecated - use run_manager
            auto_save: Save after each agent completes
            validate_after_each: Validate spec after each agent
        """
        self.run_manager = get_run_manager()
        # Auto-prune old runs on startup
        self.run_manager.auto_prune()
        
        self.auto_save = auto_save
        self.validate_after_each = validate_after_each
        
        self.agents: List[AgentDefinition] = []
        self.current_spec: Optional[DatasetSpec] = None
        
        logger.info(f"Orchestrator initialized with RunManager")
    
    def register_agent(self, agent: AgentDefinition):
        """Register an agent to the pipeline."""
        self.agents.append(agent)
        logger.info(f"Registered agent: {agent.name}")
    
    def run(
        self,
        dataset_name: str,
        description: str,
        resume_from: Optional[str] = None
    ) -> DatasetSpec:
        """
        Run the full pipeline.
        
        Args:
            dataset_name: Name of dataset
            description: User description
            resume_from: Optional spec_id to resume from
            
        Returns:
            Final DatasetSpec
        """
        logger.info("="*70)
        logger.info("PIPELINE EXECUTION STARTED")
        logger.info("="*70)
        
        # Create or load spec
        if resume_from:
            logger.info(f"Resuming from spec: {resume_from}")
            spec = self.run_manager.load_spec(resume_from)
            if spec is None:
                raise ValueError(f"Could not load spec: {resume_from}")
        else:
            logger.info(f"Creating new spec: {dataset_name}")
            from .dataset_spec import create_initial_spec
            spec = create_initial_spec(dataset_name, description)
        
        self.current_spec = spec
        spec.pipeline_status = PipelineStatus.IN_PROGRESS
        
        # Save initial state
        if self.auto_save and not resume_from:
            self.run_manager.save_spec(spec)
        
        # Run each agent
        for agent_def in self.agents:
            try:
                spec = self._run_agent(spec, agent_def)
                
                # REJECTION: request is outside the supported scope.
                # Mark the spec REJECTED but keep looping: every later agent with
                # skip_if_unsupported=True is recorded as SKIPPED (see _run_agent),
                # and the final-status step below never turns REJECTED into SUCCESS.
                if spec.requirement and not spec.requirement.is_supported:
                    if spec.pipeline_status != PipelineStatus.REJECTED:
                        logger.info(
                            f"Request rejected after {agent_def.name}: "
                            f"{spec.requirement.unsupported_reason}"
                        )
                    spec.pipeline_status = PipelineStatus.REJECTED
                    if spec.current_stage in ("initialization", ""):
                        spec.current_stage = "requirement_rejected"
                    continue
                
                # PAUSE LOGIC: After RequirementAnalyzer, check if we need user clarification
                if agent_def.name == "RequirementAnalyzer" and spec.requirement:
                    if self._should_pause_for_clarification(spec.requirement):
                        logger.warning("Pipeline paused: Requirement needs clarification")
                        spec.pipeline_status = PipelineStatus.PAUSED
                        spec.add_warning(
                            "Pipeline paused for user clarification. "
                            f"Completeness: {spec.requirement.completeness:.2f}, "
                            f"Questions: {len(spec.requirement.clarifying_questions)}"
                        )
                        if self.auto_save:
                            self.run_manager.save_spec(spec)
                        break  # Stop pipeline, return spec with questions
                
            except Exception as e:
                logger.error(f"Fatal error in agent {agent_def.name}: {e}")
                spec.pipeline_status = PipelineStatus.FAILED
                spec.add_error(f"Fatal error in {agent_def.name}: {e}")
                if self.auto_save:
                    self.run_manager.save_spec(spec)
                raise
        
        # Final status (only an IN_PROGRESS pipeline becomes SUCCESS;
        # REJECTED, PAUSED and FAILED are kept as they are)
        if spec.pipeline_status == PipelineStatus.IN_PROGRESS:
            spec.pipeline_status = PipelineStatus.SUCCESS
        
        # Final save
        if self.auto_save:
            self.run_manager.save_spec(spec)
        
        logger.info("="*70)
        logger.info(f"PIPELINE EXECUTION COMPLETE: {spec.pipeline_status.value}")
        logger.info("="*70)
        
        return spec
    
    def _should_pause_for_clarification(self, requirement: 'RequirementSpec') -> bool:
        """
        Check if pipeline should pause for user clarification.
        
        Pause ONLY when something essential is missing:
        - No identifiable domain/topic (completeness extremely low < 0.15)
        - Request cannot be understood at all
        
        For missing geography, size, freshness or features:
        - Continue with defaults (already in RequirementSpec)
        - Questions are kept in spec for user to see
        
        Args:
            requirement: RequirementSpec to check
            
        Returns:
            True if should pause for essential clarification
        """
        # Don't pause for unsupported requests - they're clear rejections
        if not requirement.is_supported:
            return False
        
        # Pause ONLY if no domain can be identified (request is unintelligible)
        if not requirement.domain:
            logger.info("Pausing: Request unintelligible (no domain identified)")
            return True
        
        # Otherwise continue - defaults will be used for missing optional info
        return False
    
    def _collect_agent_stats(self, agent_def: AgentDefinition) -> Dict[str, Any]:
        """
        Get {'llm_calls', 'llm_seconds', 'user_wait_seconds'} from an agent.
        
        Uses agent_def.stats_func if given, otherwise the `stats` dict on the
        object that owns run_func (e.g. RequirementAnalyzer.stats).
        Never raises: missing stats just means zeros.
        """
        try:
            if agent_def.stats_func is not None:
                return dict(agent_def.stats_func() or {})
            owner = getattr(agent_def.run_func, "__self__", None)
            stats = getattr(owner, "stats", None)
            if isinstance(stats, dict):
                return dict(stats)
        except Exception as e:
            logger.warning(f"Could not read stats for {agent_def.name}: {e}")
        return {}
    
    def _apply_agent_stats(self, execution: AgentExecution, stats: Dict[str, Any]):
        """Copy agent stats into the AgentExecution record."""
        if not stats:
            return
        execution.llm_calls = int(stats.get("llm_calls", 0) or 0)
        # These two fields exist after the dataset_spec.py update; guard so an
        # older dataset_spec.py does not crash the pipeline.
        if hasattr(execution, "llm_seconds"):
            execution.llm_seconds = float(stats.get("llm_seconds", 0.0) or 0.0)
        if hasattr(execution, "user_wait_seconds"):
            execution.user_wait_seconds = float(stats.get("user_wait_seconds", 0.0) or 0.0)
    
    def _run_agent(self, spec: DatasetSpec, agent_def: AgentDefinition) -> DatasetSpec:
        """Run a single agent."""
        logger.info(f"\n{'='*70}")
        logger.info(f"AGENT: {agent_def.name}")
        logger.info(f"{'='*70}")
        
        # Check if already completed (for resume)
        if spec.has_agent_succeeded(agent_def.name):
            logger.info(f"Agent {agent_def.name} already completed successfully. Skipping.")
            return spec
        
        # Check skip conditions
        if agent_def.skip_if_unsupported:
            if spec.requirement and not spec.requirement.is_supported:
                logger.info(f"Requirement unsupported. Skipping {agent_def.name}.")
                exec = AgentExecution(
                    agent_name=agent_def.name,
                    status=AgentStatus.SKIPPED,
                    started_at=datetime.now().isoformat(),
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=0.0
                )
                spec.add_execution(exec)
                return spec
        
        # Check required inputs
        for required_input in agent_def.required_inputs:
            if not hasattr(spec, required_input) or getattr(spec, required_input) is None:
                error_msg = f"Agent {agent_def.name} requires '{required_input}' but it's missing"
                logger.error(error_msg)
                spec.add_error(error_msg)
                
                exec = AgentExecution(
                    agent_name=agent_def.name,
                    status=AgentStatus.FAILED,
                    started_at=datetime.now().isoformat(),
                    completed_at=datetime.now().isoformat(),
                    duration_seconds=0.0,
                    error_message=error_msg
                )
                spec.add_execution(exec)
                spec.pipeline_status = PipelineStatus.FAILED
                return spec
        
        # Run agent
        start_time = datetime.now()
        exec = AgentExecution(
            agent_name=agent_def.name,
            status=AgentStatus.RUNNING,
            started_at=start_time.isoformat(),
            outputs_written=agent_def.outputs
        )
        spec.add_execution(exec)
        
        try:
            logger.info(f"Running {agent_def.name}...")
            
            # Execute agent
            updated_spec = agent_def.run_func(spec)
            
            # Update execution record
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            exec.status = AgentStatus.SUCCESS
            exec.completed_at = end_time.isoformat()
            exec.duration_seconds = duration
            
            # LLM calls / LLM time / time spent waiting for the user
            stats = self._collect_agent_stats(agent_def)
            self._apply_agent_stats(exec, stats)
            
            # Update in spec (replace last execution)
            updated_spec.execution_history[-1] = exec
            
            logger.info(f"✅ {agent_def.name} completed in {duration:.2f}s")
            if stats:
                logger.info(
                    f"   LLM: {exec.llm_calls} call(s), "
                    f"{float(stats.get('llm_seconds', 0.0) or 0.0):.2f}s | "
                    f"user wait: {float(stats.get('user_wait_seconds', 0.0) or 0.0):.2f}s"
                )
            
            # Validate if enabled
            if self.validate_after_each:
                is_valid, errors = validate_spec(updated_spec)
                if not is_valid:
                    logger.warning(f"Validation warnings after {agent_def.name}:")
                    for error in errors:
                        logger.warning(f"  - {error}")
                        updated_spec.add_warning(f"{agent_def.name}: {error}")
            
            # Save if enabled
            if self.auto_save:
                self.run_manager.save_spec(updated_spec)
            
            return updated_spec
            
        except Exception as e:
            # Update execution record
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            exec.status = AgentStatus.FAILED
            exec.completed_at = end_time.isoformat()
            exec.duration_seconds = duration
            exec.error_message = str(e)
            self._apply_agent_stats(exec, self._collect_agent_stats(agent_def))
            
            # Update in spec
            spec.execution_history[-1] = exec
            
            logger.error(f"❌ {agent_def.name} failed after {duration:.2f}s: {e}")
            spec.add_error(f"{agent_def.name}: {e}")
            
            # Save failure state
            if self.auto_save:
                self.run_manager.save_spec(spec)
            
            # Re-raise to stop pipeline
            raise
    
    def save_spec(self, spec: DatasetSpec) -> Path:
        """
        Save spec using run manager.
        
        Args:
            spec: Spec to save
            
        Returns:
            Path where saved
        """
        return self.run_manager.save_spec(spec)
    
    def load_spec(self, spec_id: str) -> Optional[DatasetSpec]:
        """
        Load spec using run manager.
        
        Args:
            spec_id: ID of spec to load
            
        Returns:
            Loaded spec or None if not found
        """
        return self.run_manager.load_spec(spec_id)
    
    def list_specs(self) -> List[str]:
        """List all saved spec IDs."""
        runs = self.run_manager.list_runs()
        return [spec_id for spec_id, _, _, _ in runs]
    
    def get_current_spec(self) -> Optional[DatasetSpec]:
        """Get current spec being processed."""
        return self.current_spec


# Convenience function
def create_orchestrator(save_dir: str = "./runs") -> Orchestrator:
    """Create a new orchestrator instance."""
    return Orchestrator()