"""
DataSmith AI - Pipeline Orchestrator

Runs agents in order, passes DatasetSpec through pipeline,
saves after each step, supports resume from any point.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime

from .dataset_spec import (
    DatasetSpec,
    AgentExecution,
    AgentStatus,
    PipelineStatus,
    validate_spec
)


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
        skip_if_unsupported: bool = True
    ):
        """
        Define an agent.
        
        Args:
            name: Agent name (e.g., "RequirementAnalyzer")
            run_func: Function that takes spec and returns updated spec
            required_inputs: Which spec sections must exist before this runs
            outputs: Which spec sections this agent writes
            skip_if_unsupported: Skip if requirement is unsupported
        """
        self.name = name
        self.run_func = run_func
        self.required_inputs = required_inputs
        self.outputs = outputs
        self.skip_if_unsupported = skip_if_unsupported


class Orchestrator:
    """
    Pipeline orchestrator.
    
    Responsibilities:
    - Run agents in order
    - Pass DatasetSpec between agents
    - Save spec after each agent (for resume)
    - Handle errors and skip conditions
    - Comprehensive logging
    
    Usage:
        orchestrator = Orchestrator(save_dir="./runs")
        orchestrator.register_agent(agent_def)
        spec = orchestrator.run(dataset_name, description)
    """
    
    def __init__(
        self,
        save_dir: str = "./runs",
        auto_save: bool = True,
        validate_after_each: bool = True
    ):
        """
        Initialize orchestrator.
        
        Args:
            save_dir: Directory to save specs (for resume)
            auto_save: Save after each agent completes
            validate_after_each: Validate spec after each agent
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.validate_after_each = validate_after_each
        
        self.agents: List[AgentDefinition] = []
        self.current_spec: Optional[DatasetSpec] = None
        
        logger.info(f"Orchestrator initialized (save_dir={save_dir})")
    
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
            spec = self.load_spec(resume_from)
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
            self.save_spec(spec)
        
        # Run each agent
        for agent_def in self.agents:
            try:
                spec = self._run_agent(spec, agent_def)
            except Exception as e:
                logger.error(f"Fatal error in agent {agent_def.name}: {e}")
                spec.pipeline_status = PipelineStatus.FAILED
                spec.add_error(f"Fatal error in {agent_def.name}: {e}")
                if self.auto_save:
                    self.save_spec(spec)
                raise
        
        # Final status
        if spec.pipeline_status == PipelineStatus.IN_PROGRESS:
            spec.pipeline_status = PipelineStatus.SUCCESS
        
        # Final save
        if self.auto_save:
            self.save_spec(spec)
        
        logger.info("="*70)
        logger.info(f"PIPELINE EXECUTION COMPLETE: {spec.pipeline_status.value}")
        logger.info("="*70)
        
        return spec
    
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
            
            # Update in spec (replace last execution)
            updated_spec.execution_history[-1] = exec
            
            logger.info(f"✅ {agent_def.name} completed in {duration:.2f}s")
            
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
                self.save_spec(updated_spec)
            
            return updated_spec
            
        except Exception as e:
            # Update execution record
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            exec.status = AgentStatus.FAILED
            exec.completed_at = end_time.isoformat()
            exec.duration_seconds = duration
            exec.error_message = str(e)
            
            # Update in spec
            spec.execution_history[-1] = exec
            
            logger.error(f"❌ {agent_def.name} failed after {duration:.2f}s: {e}")
            spec.add_error(f"{agent_def.name}: {e}")
            
            # Save failure state
            if self.auto_save:
                self.save_spec(spec)
            
            # Re-raise to stop pipeline
            raise
    
    def save_spec(self, spec: DatasetSpec) -> Path:
        """
        Save spec to disk.
        
        Args:
            spec: Spec to save
            
        Returns:
            Path where saved
        """
        spec.mark_updated()
        
        # Filename: spec_id.json
        filepath = self.save_dir / f"{spec.spec_id}.json"
        
        # Convert to JSON
        spec_dict = spec.model_dump(mode='json')
        
        # Write
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(spec_dict, f, indent=2)
        
        logger.debug(f"Saved spec to: {filepath}")
        return filepath
    
    def load_spec(self, spec_id: str) -> Optional[DatasetSpec]:
        """
        Load spec from disk.
        
        Args:
            spec_id: ID of spec to load
            
        Returns:
            Loaded spec or None if not found
        """
        filepath = self.save_dir / f"{spec_id}.json"
        
        if not filepath.exists():
            logger.error(f"Spec not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                spec_dict = json.load(f)
            
            spec = DatasetSpec(**spec_dict)
            logger.info(f"Loaded spec from: {filepath}")
            return spec
            
        except Exception as e:
            logger.error(f"Failed to load spec: {e}")
            return None
    
    def list_specs(self) -> List[str]:
        """List all saved spec IDs."""
        spec_files = list(self.save_dir.glob("*.json"))
        spec_ids = [f.stem for f in spec_files]
        return sorted(spec_ids)
    
    def get_current_spec(self) -> Optional[DatasetSpec]:
        """Get current spec being processed."""
        return self.current_spec


# Convenience function
def create_orchestrator(save_dir: str = "./runs") -> Orchestrator:
    """Create a new orchestrator instance."""
    return Orchestrator(save_dir=save_dir)
