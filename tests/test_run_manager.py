"""
Unit tests for RunManager and Retention Pruning Policy
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from core.run_manager import RunManager, RunManagerConfig
from core.dataset_spec import DatasetSpec, RequirementSpec, PipelineStatus, DataModality


@pytest.fixture
def temp_run_manager(tmp_path):
    """Provide a RunManager instance backed by pytest's tmp_path."""
    config = RunManagerConfig(
        keep_last=3,
        resumable_max_age_days=7,
        runs_dir=str(tmp_path / "runs")
    )
    return RunManager(config)


def test_save_spec_creates_current_and_spec_json(temp_run_manager):
    """Test that saving a spec creates both DS_<id>.json and current.json."""
    spec = DatasetSpec(
        spec_id="DS_20261004_120000",
        pipeline_status=PipelineStatus.IN_PROGRESS,
        requirement=RequirementSpec(
            dataset_name="Test Spec",
            description="Test Description",
            domain="environmental",
            data_modality=DataModality.TABULAR,
            completeness=0.8
        )
    )
    
    saved_path = temp_run_manager.save_spec(spec)
    assert saved_path.exists()
    assert saved_path.name == "DS_20261004_120000.json"
    
    current_path = temp_run_manager.runs_dir / "current.json"
    assert current_path.exists()
    
    with open(current_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data["spec_id"] == "DS_20261004_120000"


def test_pruning_retention_policy(temp_run_manager):
    """
    Test pruning policy:
    - Keeps 3 newest runs + current.json
    - Keeps pending run < 7 days old
    - Deletes finished/failed runs older than 3 newest
    """
    runs_dir = temp_run_manager.runs_dir
    now = datetime.now()
    
    test_runs = [
        ("DS_20261004_000001", now - timedelta(days=10), PipelineStatus.SUCCESS),
        ("DS_20261004_000002", now - timedelta(days=9), PipelineStatus.FAILED),
        ("DS_20261004_000003", now - timedelta(days=8), PipelineStatus.PENDING),
        ("DS_20261004_000004", now - timedelta(days=5), PipelineStatus.PENDING),
        ("DS_20261004_000005", now - timedelta(days=3), PipelineStatus.SUCCESS),
        ("DS_20261004_000006", now - timedelta(days=2), PipelineStatus.SUCCESS),
        ("DS_20261004_000007", now - timedelta(days=1), PipelineStatus.SUCCESS),
    ]
    
    for spec_id, dt, status in test_runs:
        spec = DatasetSpec(
            spec_id=spec_id,
            created_at=dt.isoformat(),
            updated_at=dt.isoformat(),
            pipeline_status=status,
            requirement=RequirementSpec(
                dataset_name=f"Dataset {spec_id}",
                description="Test",
                domain="environmental",
                completeness=0.5
            )
        )
        filepath = runs_dir / f"{spec_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(spec.model_dump(mode='json'), f, indent=2, default=str)
    
    (runs_dir / "current.json").write_text('{"spec_id": "current"}', encoding='utf-8')
    
    kept, deleted = temp_run_manager.prune_runs(dry_run=False)
    
    assert "current.json" in kept
    assert "DS_20261004_000007.json" in kept
    assert "DS_20261004_000006.json" in kept
    assert "DS_20261004_000005.json" in kept
    assert "DS_20261004_000004.json" in kept
    
    assert "DS_20261004_000001.json" in deleted
    assert "DS_20261004_000002.json" in deleted
    assert "DS_20261004_000003.json" in deleted


def test_safety_never_delete_outside_runs_dir(temp_run_manager, tmp_path):
    """
    Test safety guardrails:
    Prune runs must NEVER touch files outside runs_dir or files in runs_dir not matching DS_*.json.
    Exempted items tested: main.py, README.md, test_interactive_flood.py, flood_inputs.txt, test_runs/, .git/
    """
    runs_dir = temp_run_manager.runs_dir
    root_dir = tmp_path
    
    # Create files in root (outside runs_dir)
    root_main = root_dir / "main.py"
    root_main.write_text("# main.py", encoding='utf-8')
    
    root_flood_script = root_dir / "test_interactive_flood.py"
    root_flood_script.write_text("# test_interactive_flood.py", encoding='utf-8')
    
    root_flood_txt = root_dir / "flood_inputs.txt"
    root_flood_txt.write_text("sample input", encoding='utf-8')
    
    test_runs_folder = root_dir / "test_runs"
    test_runs_folder.mkdir(exist_ok=True)
    (test_runs_folder / "DS_dummy.json").write_text("{}", encoding='utf-8')
    
    git_folder = root_dir / ".git"
    git_folder.mkdir(exist_ok=True)
    (git_folder / "HEAD").write_text("ref: refs/heads/main", encoding='utf-8')
    
    # Create non-matching files inside runs_dir
    non_ds_json = runs_dir / "custom_config.json"
    non_ds_json.write_text('{"key": "value"}', encoding='utf-8')
    
    # Run auto_prune and prune_runs
    temp_run_manager.auto_prune()
    kept, deleted = temp_run_manager.prune_runs(dry_run=False)
    
    # Assert all files and directories outside runs/ are completely untouched
    assert root_main.exists()
    assert root_flood_script.exists()
    assert root_flood_txt.exists()
    assert test_runs_folder.exists()
    assert (test_runs_folder / "DS_dummy.json").exists()
    assert git_folder.exists()
    assert (git_folder / "HEAD").exists()
    
    # Assert non DS_*.json file inside runs/ is untouched
    assert non_ds_json.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

