"""
Tests for Orchestrator
"""

import pytest
import shutil
from pathlib import Path
from core.orchestrator import Orchestrator, AgentDefinition
from core.dataset_spec import DatasetSpec, RequirementSpec, DataModality, AgentStatus


def test_orchestrator_initialization():
    """Test orchestrator initialization."""
    save_dir = "./test_runs"
    orch = Orchestrator(save_dir=save_dir)
    
    assert orch.save_dir == Path(save_dir)
    assert orch.auto_save == True
    assert len(orch.agents) == 0
    
    # Cleanup
    if Path(save_dir).exists():
        shutil.rmtree(save_dir)


def test_agent_registration():
    """Test registering agents."""
    orch = Orchestrator(save_dir="./test_runs")
    
    def dummy_agent(spec: DatasetSpec) -> DatasetSpec:
        return spec
    
    agent = AgentDefinition(
        name="TestAgent",
        run_func=dummy_agent,
        required_inputs=[],
        outputs=["requirement"]
    )
    
    orch.register_agent(agent)
    assert len(orch.agents) == 1
    assert orch.agents[0].name == "TestAgent"
    
    # Cleanup
    if Path("./test_runs").exists():
        shutil.rmtree("./test_runs")


def test_save_and_load_spec():
    """Test saving and loading specs."""
    save_dir = "./test_runs"
    orch = Orchestrator(save_dir=save_dir)
    
    from core.dataset_spec import create_initial_spec
    spec = create_initial_spec("Test", "Test description")
    
    # Save
    filepath = orch.save_spec(spec)
    assert filepath.exists()
    
    # Load
    loaded_spec = orch.load_spec(spec.spec_id)
    assert loaded_spec is not None
    assert loaded_spec.spec_id == spec.spec_id
    assert loaded_spec.requirement.dataset_name == "Test"
    
    # Cleanup
    shutil.rmtree(save_dir)


def test_simple_pipeline():
    """Test running a simple pipeline."""
    save_dir = "./test_runs"
    orch = Orchestrator(save_dir=save_dir, auto_save=True)
    
    # Define a simple agent that adds requirement
    def requirement_agent(spec: DatasetSpec) -> DatasetSpec:
        spec.requirement = RequirementSpec(
            dataset_name=spec.requirement.dataset_name,
            description=spec.requirement.description,
            domain="test_domain",
            data_modality=DataModality.TABULAR,
            confidence=0.9,
            is_supported=True
        )
        return spec
    
    agent = AgentDefinition(
        name="RequirementAgent",
        run_func=requirement_agent,
        required_inputs=[],
        outputs=["requirement"]
    )
    
    orch.register_agent(agent)
    
    # Run pipeline
    result = orch.run("Test Dataset", "A test description")
    
    # Verify
    assert result.requirement is not None
    assert result.requirement.domain == "test_domain"
    assert len(result.execution_history) == 1
    assert result.execution_history[0].agent_name == "RequirementAgent"
    assert result.execution_history[0].status == AgentStatus.SUCCESS
    
    # Cleanup
    shutil.rmtree(save_dir)


def test_skip_on_unsupported():
    """Test skipping agents when requirement unsupported."""
    save_dir = "./test_runs"
    orch = Orchestrator(save_dir=save_dir, auto_save=False)
    
    # Agent 1: marks requirement as unsupported
    def requirement_agent(spec: DatasetSpec) -> DatasetSpec:
        spec.requirement = RequirementSpec(
            dataset_name=spec.requirement.dataset_name,
            description=spec.requirement.description,
            is_supported=False,
            unsupported_reason="Test unsupported",
            confidence=0.0
        )
        return spec
    
    # Agent 2: should be skipped
    def planner_agent(spec: DatasetSpec) -> DatasetSpec:
        raise Exception("This should not run!")
    
    agent1 = AgentDefinition(
        name="RequirementAgent",
        run_func=requirement_agent,
        required_inputs=[],
        outputs=["requirement"],
        skip_if_unsupported=False  # Don't skip this one
    )
    
    agent2 = AgentDefinition(
        name="PlannerAgent",
        run_func=planner_agent,
        required_inputs=["requirement"],
        outputs=["plan"],
        skip_if_unsupported=True  # Skip if unsupported
    )
    
    orch.register_agent(agent1)
    orch.register_agent(agent2)
    
    # Run
    result = orch.run("Test", "Test")
    
    # Verify
    assert len(result.execution_history) == 2
    assert result.execution_history[0].status == AgentStatus.SUCCESS
    assert result.execution_history[1].status == AgentStatus.SKIPPED
    
    # Cleanup
    if Path(save_dir).exists():
        shutil.rmtree(save_dir)


def test_list_specs():
    """Test listing saved specs."""
    save_dir = "./test_runs"
    orch = Orchestrator(save_dir=save_dir)
    
    from core.dataset_spec import create_initial_spec
    import time
    
    # Save multiple specs (with small delay to ensure different IDs)
    spec1 = create_initial_spec("Test1", "Desc1")
    orch.save_spec(spec1)
    
    time.sleep(1.1)  # Ensure different timestamp
    
    spec2 = create_initial_spec("Test2", "Desc2")
    orch.save_spec(spec2)
    
    # List
    specs = orch.list_specs()
    assert len(specs) == 2
    assert spec1.spec_id in specs
    assert spec2.spec_id in specs
    
    # Cleanup
    shutil.rmtree(save_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
