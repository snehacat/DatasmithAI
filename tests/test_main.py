import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from core.dataset_spec import DatasetSpec, PipelineStatus, RequirementSpec
from core.orchestrator import AgentDefinition


def test_setup_logging_writes_to_file(tmp_path):
    root = logging.getLogger()
    old_level = root.level
    handler = main.setup_logging(log_dir=tmp_path)
    try:
        logging.getLogger("datasmith.test").info("hello log")
        handler.flush()
        text = (tmp_path / "datasmith.log").read_text(encoding="utf-8")
        assert "hello log" in text
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(old_level)


def test_main_registers_planning_agent_after_requirement_analyzer(monkeypatch):
    registered = []
    orchestrator_args = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            orchestrator_args.update(kwargs)

        def register_agent(self, agent):
            registered.append(agent)

        def run(self, dataset_name, description):
            return DatasetSpec(
                spec_id="DS_test",
                pipeline_status=PipelineStatus.REJECTED,
                requirement=RequirementSpec(
                    dataset_name=dataset_name,
                    description=description,
                    is_supported=False,
                    unsupported_reason="Not supported",
                ),
            )

    analyzer = SimpleNamespace(analyze=lambda spec: spec)
    planner = SimpleNamespace(plan=lambda spec: spec)
    monkeypatch.setattr(main, "setup_logging", Mock())
    monkeypatch.setattr(main, "warm_up_ollama", Mock())
    monkeypatch.setattr(main, "RequirementAnalyzer", Mock(return_value=analyzer))
    monkeypatch.setattr(main, "PlanningAgent", Mock(return_value=planner))
    monkeypatch.setattr(main, "get_run_manager", Mock(return_value=Mock()))
    monkeypatch.setattr("core.orchestrator.Orchestrator", FakeOrchestrator)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    inputs = iter(["Weather", "weather data for Delhi"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    main.main()

    assert orchestrator_args == {"auto_prune": True}
    assert [agent.name for agent in registered] == ["RequirementAnalyzer", "PlanningAgent"]
    plan_agent = registered[1]
    assert isinstance(plan_agent, AgentDefinition)
    assert plan_agent.run_func == planner.plan
    assert plan_agent.required_inputs == ["requirement"]
    assert plan_agent.outputs == ["plan"]
    assert plan_agent.skip_if_unsupported is True