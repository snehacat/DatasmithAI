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


def test_load_spec_rejects_parent_traversal(temp_run_manager):
    assert temp_run_manager.load_spec("../../x") is None


def test_pruning_never_deletes_current(temp_run_manager):
    (temp_run_manager.runs_dir / "current.json").write_text("{}", encoding="utf-8")
    kept, _ = temp_run_manager.prune_runs(dry_run=False)
    assert "current.json" in kept


def test_old_rejected_run_is_deleted(temp_run_manager):
    runs_dir = temp_run_manager.runs_dir
    spec_id = "DS_old_rejected"
    old = datetime.now() - timedelta(days=2)
    spec = DatasetSpec(
        spec_id=spec_id,
        created_at=old.isoformat(),
        updated_at=old.isoformat(),
        pipeline_status=PipelineStatus.REJECTED,
        requirement=RequirementSpec(dataset_name="Rejected", description="stock"),
    )
    (runs_dir / f"{spec_id}.json").write_text(
        json.dumps(spec.model_dump(mode="json"), default=str), encoding="utf-8"
    )
    _, deleted = temp_run_manager.prune_runs(dry_run=False)
    assert f"{spec_id}.json" in deleted
    assert not (runs_dir / f"{spec_id}.json").exists()


def test_young_paused_run_is_kept(temp_run_manager):
    runs_dir = temp_run_manager.runs_dir
    spec_id = "DS_young_paused"
    recent = datetime.now() - timedelta(days=3)
    spec = DatasetSpec(
        spec_id=spec_id,
        created_at=recent.isoformat(),
        updated_at=recent.isoformat(),
        pipeline_status=PipelineStatus.PAUSED,
        requirement=RequirementSpec(dataset_name="Paused", description="vague"),
    )
    (runs_dir / f"{spec_id}.json").write_text(
        json.dumps(spec.model_dump(mode="json"), default=str), encoding="utf-8"
    )
    kept, _ = temp_run_manager.prune_runs(dry_run=False)
    assert f"{spec_id}.json" in kept


@pytest.mark.parametrize(
    "spec_id",
    ["../outside", "DS_../outside", "C:\\outside", "not_a_spec_id", "DS_"],
)
def test_raw_artifact_directory_rejects_invalid_spec_ids(temp_run_manager, spec_id):
    with pytest.raises(ValueError):
        temp_run_manager.get_raw_artifact_dir(spec_id)


def test_raw_artifact_directory_is_created_under_artifacts_root(temp_run_manager):
    spec_id = "DS_20261007_123000_abcdef"

    raw_dir = temp_run_manager.get_raw_artifact_dir(spec_id)

    assert raw_dir.is_dir()
    assert raw_dir == temp_run_manager.runs_dir / "artifacts" / spec_id / "raw"


def _write_old_rejected_run(run_manager, spec_id):
    old = datetime.now() - timedelta(days=2)
    spec = DatasetSpec(
        spec_id=spec_id,
        created_at=old.isoformat(),
        updated_at=old.isoformat(),
        pipeline_status=PipelineStatus.REJECTED,
        requirement=RequirementSpec(dataset_name="Rejected", description="old request"),
    )
    (run_manager.runs_dir / f"{spec_id}.json").write_text(
        json.dumps(spec.model_dump(mode="json"), default=str), encoding="utf-8"
    )


def test_pruning_removes_artifacts_for_pruned_run(temp_run_manager):
    spec_id = "DS_old_with_artifacts"
    raw_dir = temp_run_manager.get_raw_artifact_dir(spec_id)
    (raw_dir / "data.csv").write_text("value\n1\n", encoding="utf-8")
    _write_old_rejected_run(temp_run_manager, spec_id)

    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert not (temp_run_manager.runs_dir / f"artifacts" / spec_id).exists()


def test_pruning_preserves_artifacts_for_unpruned_and_unexpected_directories(temp_run_manager):
    old_spec_id = "DS_old_artifacts"
    kept_spec_id = "DS_kept_artifacts"
    old_raw = temp_run_manager.get_raw_artifact_dir(old_spec_id)
    kept_raw = temp_run_manager.get_raw_artifact_dir(kept_spec_id)
    (old_raw / "old.csv").write_text("old", encoding="utf-8")
    (kept_raw / "kept.csv").write_text("kept", encoding="utf-8")
    _write_old_rejected_run(temp_run_manager, old_spec_id)

    kept_spec = DatasetSpec(
        spec_id=kept_spec_id,
        pipeline_status=PipelineStatus.REJECTED,
        requirement=RequirementSpec(dataset_name="Recent rejection", description="recent"),
    )
    (temp_run_manager.runs_dir / f"{kept_spec_id}.json").write_text(
        json.dumps(kept_spec.model_dump(mode="json"), default=str), encoding="utf-8"
    )

    unexpected_dir = temp_run_manager.runs_dir / "artifacts" / "not_a_spec"
    unexpected_dir.mkdir()
    marker = unexpected_dir / "preserve.txt"
    marker.write_text("unrelated", encoding="utf-8")
    malformed_dir = temp_run_manager.runs_dir / "artifacts" / "DS_bad.name"
    malformed_dir.mkdir()

    _, deleted = temp_run_manager.prune_runs()

    assert f"{old_spec_id}.json" in deleted
    assert not (temp_run_manager.runs_dir / "artifacts" / old_spec_id).exists()
    assert (kept_raw / "kept.csv").exists()
    assert marker.exists()
    assert malformed_dir.exists()


def test_pruning_handles_missing_artifact_directory(temp_run_manager):
    spec_id = "DS_old_without_artifacts"
    _write_old_rejected_run(temp_run_manager, spec_id)

    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted


def test_pruning_does_not_follow_artifact_root_outside_runs(temp_run_manager, tmp_path):
    outside_artifacts = tmp_path / "outside_artifacts"
    outside_run = outside_artifacts / "DS_old_with_artifacts"
    outside_raw = outside_run / "raw"
    outside_raw.mkdir(parents=True)
    marker = outside_raw / "preserve.txt"
    marker.write_text("outside", encoding="utf-8")

    artifact_root_link = temp_run_manager.runs_dir / "artifacts"
    try:
        artifact_root_link.symlink_to(outside_artifacts, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    spec_id = "DS_old_with_artifacts"
    _write_old_rejected_run(temp_run_manager, spec_id)

    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert marker.exists()
    assert outside_raw.is_dir()


def test_pruning_skips_artifact_root_resolved_outside_runs(temp_run_manager, tmp_path, monkeypatch):
    outside_artifacts = tmp_path / "resolved_outside_artifacts"
    outside_run = outside_artifacts / "DS_old_with_artifacts"
    outside_raw = outside_run / "raw"
    outside_raw.mkdir(parents=True)
    marker = outside_raw / "preserve.txt"
    marker.write_text("outside", encoding="utf-8")
    spec_id = "DS_old_with_artifacts"
    _write_old_rejected_run(temp_run_manager, spec_id)

    artifact_root = temp_run_manager.runs_dir / "artifacts"
    original_resolve = Path.resolve

    def resolve_with_external_artifact_root(path, strict=False):
        if path == artifact_root:
            return outside_artifacts
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_with_external_artifact_root)

    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert marker.exists()


def test_pruning_rejects_artifact_root_resolving_to_sibling_inside_runs(
    temp_run_manager, monkeypatch
):
    spec_id = "DS_run_A"
    artifacts_root = temp_run_manager.runs_dir / "artifacts"
    artifacts_root.mkdir()
    sibling_root = temp_run_manager.runs_dir / "artifacts_alias"
    target_raw = sibling_root / spec_id / "raw"
    target_raw.mkdir(parents=True)
    marker = target_raw / "preserve.txt"
    marker.write_text("unrelated", encoding="utf-8")
    _write_old_rejected_run(temp_run_manager, spec_id)

    original_resolve = Path.resolve

    def resolve_artifacts_to_sibling(path, strict=False):
        if path == artifacts_root:
            return sibling_root
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_artifacts_to_sibling)

    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert marker.exists()
    assert target_raw.is_dir()


def test_pruning_rejects_spec_directory_resolving_to_sibling_run(
    temp_run_manager, monkeypatch
):
    spec_id = "DS_run_A"
    artifacts_root = temp_run_manager.runs_dir / "artifacts"
    spec_dir = artifacts_root / spec_id
    sibling_spec_dir = artifacts_root / "DS_run_B"
    spec_dir.mkdir(parents=True)
    target_raw = sibling_spec_dir / "raw"
    target_raw.mkdir(parents=True)
    marker = target_raw / "preserve.txt"
    marker.write_text("unrelated", encoding="utf-8")
    _write_old_rejected_run(temp_run_manager, spec_id)

    original_resolve = Path.resolve

    def resolve_spec_to_sibling(path, strict=False):
        if path == spec_dir:
            return sibling_spec_dir
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_spec_to_sibling)

    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert spec_dir.is_dir()
    assert marker.exists()
    assert target_raw.is_dir()


def test_raw_artifact_directory_rejects_resolution_to_sibling_inside_spec(
    temp_run_manager, monkeypatch
):
    spec_id = "DS_run_A"
    spec_dir = temp_run_manager.runs_dir / "artifacts" / spec_id
    raw_dir = spec_dir / "raw"
    alternate_raw_dir = spec_dir / "raw_alias"
    alternate_raw_dir.mkdir(parents=True)
    marker = alternate_raw_dir / "preserve.txt"
    marker.write_text("unrelated", encoding="utf-8")

    original_resolve = Path.resolve
    raw_resolve_count = [0]

    def resolve_raw_to_sibling(path, strict=False):
        if path == raw_dir:
            raw_resolve_count[0] += 1
            if raw_resolve_count[0] == 2:
                return alternate_raw_dir
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_raw_to_sibling)

    with pytest.raises(ValueError, match="Raw artifact directory resolves"):
        temp_run_manager.get_raw_artifact_dir(spec_id)

    assert raw_resolve_count[0] == 2
    assert raw_dir.is_dir()
    assert marker.exists()
    assert alternate_raw_dir.is_dir()


def _create_directory_symlink_or_skip(target, link):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")


def test_artifact_spec_symlink_is_rejected_and_target_is_preserved(
    temp_run_manager, tmp_path, caplog
):
    spec_id = "DS_linked_spec"
    artifacts_root = temp_run_manager.runs_dir / "artifacts"
    artifacts_root.mkdir()
    outside_target = tmp_path / "outside_spec_target"
    outside_raw = outside_target / "raw"
    outside_raw.mkdir(parents=True)
    marker = outside_raw / "preserve.txt"
    marker.write_text("outside", encoding="utf-8")
    spec_link = artifacts_root / spec_id
    _create_directory_symlink_or_skip(outside_target, spec_link)

    with pytest.raises(ValueError, match="symbolic link"):
        temp_run_manager.get_raw_artifact_dir(spec_id)

    _write_old_rejected_run(temp_run_manager, spec_id)
    with caplog.at_level("WARNING", logger="core.run_manager"):
        _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert "Skipped unsafe artifact deletion" in caplog.text
    assert spec_link.is_symlink()
    assert marker.exists()


def test_raw_artifact_symlink_is_rejected_and_target_is_preserved(
    temp_run_manager, tmp_path
):
    spec_id = "DS_linked_raw"
    raw_dir = temp_run_manager.get_raw_artifact_dir(spec_id)
    raw_dir.rmdir()
    outside_raw = tmp_path / "outside_raw_target"
    outside_raw.mkdir()
    marker = outside_raw / "preserve.txt"
    marker.write_text("outside", encoding="utf-8")
    _create_directory_symlink_or_skip(outside_raw, raw_dir)

    with pytest.raises(ValueError, match="symbolic link"):
        temp_run_manager.get_raw_artifact_dir(spec_id)

    _write_old_rejected_run(temp_run_manager, spec_id)
    _, deleted = temp_run_manager.prune_runs()

    assert f"{spec_id}.json" in deleted
    assert marker.exists()
    assert not raw_dir.is_symlink()


def test_pruning_handles_artifact_deletion_failure_without_external_cleanup(
    temp_run_manager, tmp_path, monkeypatch, caplog
):
    spec_id = "DS_artifact_delete_failure"
    raw_dir = temp_run_manager.get_raw_artifact_dir(spec_id)
    (raw_dir / "keep.txt").write_text("inside", encoding="utf-8")
    _write_old_rejected_run(temp_run_manager, spec_id)
    outside_dir = tmp_path / "unrelated_outside"
    outside_dir.mkdir()
    outside_marker = outside_dir / "preserve.txt"
    outside_marker.write_text("outside", encoding="utf-8")

    attempted_paths = []

    def fail_artifact_removal(path):
        attempted_paths.append(Path(path))
        raise PermissionError("simulated deletion failure")

    monkeypatch.setattr("core.run_manager.shutil.rmtree", fail_artifact_removal)

    with caplog.at_level("ERROR", logger="core.run_manager"):
        _, deleted = temp_run_manager.prune_runs()

    artifact_dir = temp_run_manager.runs_dir / "artifacts" / spec_id
    assert f"{spec_id}.json" in deleted
    assert attempted_paths == [artifact_dir]
    assert artifact_dir.is_dir()
    assert (raw_dir / "keep.txt").exists()
    assert outside_marker.exists()
    assert "Failed to delete artifacts for" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

