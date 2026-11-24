"""
Tests for Orchestrator
"""

import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
from core.orchestrator import Orchestrator, AgentDefinition
from core.dataset_spec import DatasetSpec, RequirementSpec, DataModality, AgentStatus, PipelineStatus
from core.run_manager import RunManager, RunManagerConfig


@pytest.fixture
def temp_run_manager(tmp_path):
    """Fixture providing a RunManager backed by a temporary directory."""
    config = RunManagerConfig(runs_dir=str(tmp_path))
    return RunManager(config)


def test_orchestrator_initialization(temp_run_manager):
    """Test orchestrator initialization."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        assert orch.auto_save == True
        assert len(orch.agents) == 0
        assert orch.run_manager == temp_run_manager


def test_orchestrator_does_not_prune_by_default(temp_run_manager):
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        temp_run_manager.auto_prune = Mock()
        Orchestrator()
        temp_run_manager.auto_prune.assert_not_called()


def test_save_dir_builds_run_manager(tmp_path):
    orch = Orchestrator(save_dir=str(tmp_path / "custom-runs"))
    assert orch.run_manager.runs_dir == (tmp_path / "custom-runs").resolve()


def test_agent_registration(temp_run_manager):
    """Test registering agents."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        
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


def test_save_and_load_spec(temp_run_manager):
    """Test saving and loading specs."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        
        from core.dataset_spec import create_initial_spec
        spec = create_initial_spec("Test", "Test description")
        
        # Save
        filepath = orch.save_spec(spec)
        assert filepath.exists()
        assert (temp_run_manager.runs_dir / "current.json").exists()
        
        # Load
        loaded_spec = orch.load_spec(spec.spec_id)
        assert loaded_spec is not None
        assert loaded_spec.spec_id == spec.spec_id
        assert loaded_spec.requirement.dataset_name == "Test"


def test_simple_pipeline(temp_run_manager):
    """Test running a simple pipeline."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator(auto_save=True)
        
        def requirement_agent(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name=spec.requirement.dataset_name,
                description=spec.requirement.description,
                domain="test_domain",
                data_modality=DataModality.TABULAR,
                completeness=0.9,
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


def test_skip_on_unsupported(temp_run_manager):
    """Test skipping agents when requirement unsupported."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator(auto_save=False)
        
        def requirement_agent(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name=spec.requirement.dataset_name,
                description=spec.requirement.description,
                is_supported=False,
                unsupported_reason="Test unsupported",
                completeness=0.0
            )
            return spec
        
        def planner_agent(spec: DatasetSpec) -> DatasetSpec:
            raise Exception("This should not run!")
        
        agent1 = AgentDefinition(
            name="RequirementAgent",
            run_func=requirement_agent,
            required_inputs=[],
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        
        agent2 = AgentDefinition(
            name="PlannerAgent",
            run_func=planner_agent,
            required_inputs=["requirement"],
            outputs=["plan"],
            skip_if_unsupported=True
        )
        
        orch.register_agent(agent1)
        orch.register_agent(agent2)
        
        # Run
        result = orch.run("Test", "Test")
        
        # Verify
        assert len(result.execution_history) == 2
        assert result.execution_history[0].status == AgentStatus.SUCCESS
        assert result.execution_history[1].status == AgentStatus.SKIPPED


def test_list_specs(temp_run_manager):
    """Test listing saved specs."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        
        from core.dataset_spec import create_initial_spec
        import time
        
        spec1 = create_initial_spec("Test1", "Desc1")
        orch.save_spec(spec1)
        
        time.sleep(1.1)
        
        spec2 = create_initial_spec("Test2", "Desc2")
        orch.save_spec(spec2)
        
        specs = orch.list_specs()
        assert len(specs) == 2
        assert spec1.spec_id in specs
        assert spec2.spec_id in specs


def test_pause_for_unintelligible(temp_run_manager):
    """Test that orchestrator pauses when request is completely unintelligible."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orchestrator = Orchestrator()
        
        def mock_analyzer(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name="Unintelligible",
                description="asdfghjkl random text",
                domain=None,
                completeness=0.05,
                is_supported=True
            )
            return spec
        
        agent_def = AgentDefinition(
            name="RequirementAnalyzer",
            run_func=mock_analyzer,
            required_inputs=[],
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        orchestrator.register_agent(agent_def)
        
        spec = orchestrator.run("Test", "unintelligible request")
        
        assert spec.pipeline_status == PipelineStatus.PAUSED
        assert len(spec.warnings) > 0
        assert "paused" in spec.warnings[0].lower() or "unintelligible" in spec.warnings[0].lower()


def test_no_pause_for_vague_with_domain(temp_run_manager):
    """Test that orchestrator does NOT pause for vague requests like 'some weather data' (low completeness=0.10) if domain is set."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orchestrator = Orchestrator()
        
        def mock_analyzer(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name="Some Weather Data",
                description="some weather data",
                domain="meteorological",
                subdomain=None,
                completeness=0.10,  # Low completeness < 0.15
                clarifying_questions=["Which geographic area or location do you need?", "What time period do you need?"],
                is_supported=True
            )
            return spec
        
        agent_def = AgentDefinition(
            name="RequirementAnalyzer",
            run_func=mock_analyzer,
            required_inputs=[],
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        orchestrator.register_agent(agent_def)
        
        spec = orchestrator.run("Test Weather", "some weather data")
        
        # Must NOT pause (must be SUCCESS), and questions must remain in spec
        assert spec.pipeline_status == PipelineStatus.SUCCESS
        assert len(spec.requirement.clarifying_questions) == 2
        assert "Which geographic area" in spec.requirement.clarifying_questions[0]


def test_no_pause_for_clear_request(temp_run_manager):
    """Test that orchestrator does NOT pause for clear requests."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orchestrator = Orchestrator()
        
        def mock_analyzer(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name="Clear Request",
                description="clear data request",
                domain="environmental",
                completeness=0.8,
                clarifying_questions=[],
                is_supported=True
            )
            return spec
        
        agent_def = AgentDefinition(
            name="RequirementAnalyzer",
            run_func=mock_analyzer,
            required_inputs=[],
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        orchestrator.register_agent(agent_def)
        
        spec = orchestrator.run("Test", "clear request")
        assert spec.pipeline_status == PipelineStatus.SUCCESS


def test_no_pause_for_satellite_fire(temp_run_manager):
    """Test that satellite fire request does NOT pause (has domain and topic)."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orchestrator = Orchestrator()
        
        def mock_analyzer(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name="India Fire Detections",
                description="recent forest fire detections",
                domain="environmental",
                subdomain="forest_fire_monitoring",
                completeness=0.50,
                clarifying_questions=["How recent?"],
                is_supported=True
            )
            return spec
        
        agent_def = AgentDefinition(
            name="RequirementAnalyzer",
            run_func=mock_analyzer,
            required_inputs=[],
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        orchestrator.register_agent(agent_def)
        
        spec = orchestrator.run("Test", "satellite fire request")
        assert spec.pipeline_status == PipelineStatus.SUCCESS
        assert spec.requirement.domain == "environmental"


def test_no_pause_for_unsupported(temp_run_manager):
    """Test that orchestrator does NOT pause for unsupported requests."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orchestrator = Orchestrator()
        
        def mock_analyzer(spec: DatasetSpec) -> DatasetSpec:
            spec.requirement = RequirementSpec(
                dataset_name="Unsupported",
                description="stock prices",
                completeness=0.0,
                is_supported=False,
                unsupported_reason="Financial data not supported"
            )
            return spec
        
        agent_def = AgentDefinition(
            name="RequirementAnalyzer",
            run_func=mock_analyzer,
            required_inputs=[],
            outputs=["requirement"],
            skip_if_unsupported=False
        )
        orchestrator.register_agent(agent_def)
        
        spec = orchestrator.run("Test", "unsupported request")
        assert spec.pipeline_status == PipelineStatus.REJECTED
        assert spec.requirement.is_supported is False

def test_pipeline_stops_after_agent_failure(temp_run_manager):
    """A failed agent (missing required input) must stop later agents from running."""
    ran = []
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        orch.register_agent(AgentDefinition(
            name="NeedsPlan",
            run_func=lambda s: s,
            required_inputs=["plan"],      # plan is never set, so this agent fails
            outputs=["retrieval"],
            skip_if_unsupported=False,
        ))
        orch.register_agent(AgentDefinition(
            name="Later",
            run_func=lambda s: (ran.append("Later"), s)[1],
            required_inputs=[],
            outputs=[],
            skip_if_unsupported=False,
        ))
        spec = orch.run("t", "air quality in Delhi")

    assert spec.pipeline_status == PipelineStatus.FAILED
    assert ran == []   # "Later" must NOT have run
def test_resume_paused_run_with_clarified_description(temp_run_manager):
    """Resuming a paused run with a new description re-runs the analyzer and continues."""
    def analyzer(s):
        if "flood" in s.requirement.description:
            s.requirement.domain = "environmental"
        return s

    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        orch.register_agent(AgentDefinition(
            name="RequirementAnalyzer", run_func=analyzer,
            required_inputs=[], outputs=["requirement"], skip_if_unsupported=False,
        ))
        paused = orch.run("t", "something vague")
        assert paused.pipeline_status == PipelineStatus.PAUSED

        resumed = orch.run("", "flood data for Assam", resume_from=paused.spec_id)

    assert resumed.pipeline_status == PipelineStatus.SUCCESS
    assert resumed.requirement.domain == "environmental"


def test_resume_paused_run_without_description_raises(temp_run_manager):
    """A paused run cannot be resumed without the user's clarification."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        orch.register_agent(AgentDefinition(
            name="RequirementAnalyzer", run_func=lambda s: s,
            required_inputs=[], outputs=["requirement"], skip_if_unsupported=False,
        ))
        paused = orch.run("t", "something vague")
        assert paused.pipeline_status == PipelineStatus.PAUSED

        with pytest.raises(ValueError):
            orch.run("", "", resume_from=paused.spec_id)
def test_agent_returning_none_gives_clear_error(temp_run_manager):
    """An agent that forgets 'return spec' must fail with a clear message and a saved FAILED state."""
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        orch.register_agent(AgentDefinition(
            name="Forgetful",
            run_func=lambda s: None,
            required_inputs=[],
            outputs=[],
            skip_if_unsupported=False,
        ))
        with pytest.raises(TypeError, match="must return a DatasetSpec"):
            orch.run("t", "air quality in Delhi")

    saved = temp_run_manager.load_spec("current")
    assert saved.pipeline_status == PipelineStatus.FAILED
    assert saved.execution_history[-1].status == AgentStatus.FAILED


def test_success_sets_last_agent_as_current_stage(temp_run_manager):
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()
        def successful_requirement(spec):
            spec.requirement.domain = "environmental"
            spec.requirement.is_supported = True
            return spec
        orch.register_agent(AgentDefinition(
            name="RequirementAnalyzer",
            run_func=successful_requirement,
            required_inputs=[],
            outputs=[],
            skip_if_unsupported=False,
        ))
        spec = orch.run("Test", "rainfall in Kerala")
    assert spec.pipeline_status == PipelineStatus.SUCCESS
    assert spec.current_stage == "RequirementAnalyzer"


def test_fatal_agent_error_is_not_added_twice(temp_run_manager):
    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager):
        orch = Orchestrator()

        def fail(spec):
            raise RuntimeError("boom")

        orch.register_agent(AgentDefinition(
            name="FailingAgent",
            run_func=fail,
            required_inputs=[],
            outputs=[],
        ))
        with pytest.raises(RuntimeError):
            orch.run("Test", "rainfall")

    saved = temp_run_manager.load_spec("current")
    matching = [error for error in saved.errors if "boom" in error]
    assert len(matching) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
