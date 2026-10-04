"""
Test interactive clarification with flood example
"""

import sys
import time
import pytest
from pathlib import Path
from io import StringIO
from unittest.mock import patch
from agents.requirement_analyzer import RequirementAnalyzer
from core.dataset_spec import create_initial_spec
from core.run_manager import RunManager, RunManagerConfig


def test_interactive_flood_clarification(tmp_path):
    """Test interactive clarification for flood dataset."""
    # Use temporary directory for run manager to prevent leaving files in runs/
    config = RunManagerConfig(runs_dir=str(tmp_path))
    temp_run_manager = RunManager(config)

    input_responses = [
        "2",        # Time granularity: Daily
        "4",        # Dataset size: Custom amount
        "500 rows"  # Custom size
    ]

    def mock_input_generator():
        responses = iter(input_responses)
        def mock_input(prompt):
            try:
                response = next(responses)
                return response
            except StopIteration:
                return ""
        return mock_input

    with patch('core.orchestrator.get_run_manager', return_value=temp_run_manager), \
         patch('builtins.input', side_effect=mock_input_generator()):
        
        analyzer = RequirementAnalyzer(interactive=True)
        spec = create_initial_spec(
            "Flood Risk Prediction dataset of Rishikesh from 2020 to 2026",
            "Flood risk"
        )
        result = analyzer.analyze(spec)

    req = result.requirement

    assert req.geography == "Rishikesh"
    assert req.time_range == "2020 to 2026"
    assert req.expected_size == "500 rows"
    assert req.clarification_round == 1
    assert req.completeness >= 0.5
    assert any('granularity' in item.lower() for item in req.explicitly_stated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
