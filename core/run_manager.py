"""
Run Management System

Handles saving, loading, and retention pruning of dataset specification runs.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

from .dataset_spec import DatasetSpec, PipelineStatus

logger = logging.getLogger(__name__)


class RunManagerConfig:
    """Centralized configuration for run management and retention."""
    
    def __init__(
        self,
        keep_last: int = 3,
        resumable_max_age_days: int = 7,
        runs_dir: Optional[str] = None,
        rejected_max_age_days: int = 1
    ):
        self.keep_last = keep_last
        self.resumable_max_age_days = resumable_max_age_days
        self.rejected_max_age_days = rejected_max_age_days
        # Default: <project root>/runs, so results land in the same place no matter
        # which folder you start Python from.
        if runs_dir is None:
            runs_dir = str(Path(__file__).resolve().parent.parent / "runs")
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


class RunManager:
    """
    Manages dataset specification runs.
    
    Features:
    - Save specs with unique IDs
    - Maintain current.json as a copy of the latest spec
    - Automatic pruning (keep 3 newest real runs + resumable pending runs < 7 days old;
      rejected requests do not use up the "newest" slots and are removed after 1 day)
    - Strict safety checks (only delete DS_*.json inside runs_dir)
    """
    
    def __init__(self, config: Optional[RunManagerConfig] = None):
        """
        Initialize run manager.
        
        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or RunManagerConfig()
        self.runs_dir = self.config.runs_dir.resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RunManager initialized (dir={self.runs_dir})")
    
    def save_spec(self, spec: DatasetSpec, is_current: bool = True) -> Path:
        """
        Save a dataset specification.
        
        Args:
            spec: Specification to save
            is_current: Whether to also save as current.json
            
        Returns:
            Path to saved file
        """
        spec.mark_updated()
        
        # Save main file
        main_path = self.runs_dir / f"{spec.spec_id}.json"
        spec_dict = spec.model_dump(mode='json')
        
        self._write_json_atomic(main_path, spec_dict)
        logger.debug(f"Saved spec to: {main_path}")
        
        # Save as current.json if requested
        if is_current:
            self._write_json_atomic(self.runs_dir / "current.json", spec_dict)
            logger.debug(f"Updated current.json")
        
        return main_path
    
    @staticmethod
    def _write_json_atomic(path: Path, data: Dict[str, Any]):
        """Write to a temp file, then replace: a crash mid-write cannot corrupt the real file."""
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    
    def load_spec(self, spec_id: str) -> Optional[DatasetSpec]:
        """
        Load a dataset specification.
        
        Args:
            spec_id: ID of spec to load, or "current" for current.json
            
        Returns:
            Loaded spec or None if not found
        """
        if spec_id == "current":
            filepath = self.runs_dir / "current.json"
        else:
            filepath = self.runs_dir / f"{spec_id}.json"
        
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
            logger.error(f"Failed to load spec {spec_id}: {e}")
            return None
    
    def list_runs(self) -> List[Tuple[str, datetime, str, str]]:
        """
        List all available runs in runs_dir matching DS_*.json.
        
        Returns:
            List of (spec_id, timestamp, status, description) tuples, newest first
        """
        runs = []
        
        for json_file in self.runs_dir.glob("DS_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                spec_id = data.get('spec_id', json_file.stem)
                created_str = data.get('created_at', '1970-01-01T00:00:00')
                created_at = datetime.fromisoformat(created_str)
                
                status_raw = data.get('pipeline_status', 'unknown')
                status = status_raw.value if hasattr(status_raw, 'value') else str(status_raw)
                
                req_data = data.get('requirement', {}) or {}
                description = req_data.get('dataset_name', 'Unknown')
                
                runs.append((spec_id, created_at, status, description))
                
            except Exception as e:
                logger.warning(f"Failed to read {json_file}: {e}")
                continue
        
        # Sort by timestamp, newest first
        runs.sort(key=lambda x: x[1], reverse=True)
        return runs
    
    def prune_runs(self, dry_run: bool = False) -> Tuple[List[str], List[str]]:
        """
        Clean up old runs according to retention policy.
        
        Rules:
        - Always keep current.json
        - Keep the top N newest real runs (keep_last); rejected runs are not counted
          and are deleted after rejected_max_age_days
        - Keep resumable runs (pending/in_progress) newer than max_age_days
        - Delete finished (success), failed, or older runs beyond retention limits
        - Safety: Only delete DS_*.json files located strictly inside self.runs_dir
        
        Args:
            dry_run: If True, return what would be deleted without deleting
            
        Returns:
            (kept_files, deleted_files) lists of filenames
        """
        runs = self.list_runs()
        current_time = datetime.now()
        max_age = timedelta(days=self.config.resumable_max_age_days)
        rejected_max_age = timedelta(days=self.config.rejected_max_age_days)
        
        kept = []
        to_delete = []
        
        # Always keep current.json
        current_path = self.runs_dir / "current.json"
        if current_path.exists():
            kept.append("current.json")
        
        newest_kept = 0  # real (non-rejected) runs kept as "newest"
        
        for spec_id, timestamp, status, description in runs:
            filename = f"{spec_id}.json"
            age = current_time - timestamp
            status_lower = str(status).lower()
            
            # Rejected requests are throw-away: they must not push real runs out of
            # the "newest N" list. Keep them for a short time only.
            if status_lower == 'rejected':
                if age <= rejected_max_age:
                    kept.append(filename)
                else:
                    to_delete.append(filename)
                continue
            
            # Keep if in top N newest (real runs)
            if newest_kept < self.config.keep_last:
                kept.append(filename)
                newest_kept += 1
                continue
            
            # Keep if resumable (pending, in_progress, or paused) and young enough (< 7 days)
            if status_lower in ['pending', 'in_progress', 'paused'] and age <= max_age:
                kept.append(filename)
                continue
            
            # Otherwise mark for deletion
            to_delete.append(filename)
        
        # Perform deletion if not dry run
        actually_deleted = []
        if not dry_run:
            for filename in list(to_delete):
                filepath = (self.runs_dir / filename).resolve()
                try:
                    # STRICT SAFETY CHECKS:
                    # 1. File must be directly inside self.runs_dir
                    # 2. Filename must start with DS_ and end with .json
                    # 3. Must not be outside self.runs_dir
                    if (filepath.parent == self.runs_dir and 
                        filename.startswith('DS_') and 
                        filename.endswith('.json')):
                        
                        if filepath.exists():
                            filepath.unlink()
                            actually_deleted.append(filename)
                            logger.info(f"Deleted old run: {filename}")
                    else:
                        logger.warning(f"Skipped unsafe deletion: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")
        else:
            actually_deleted = to_delete
            
        return kept, actually_deleted
    
    def auto_prune(self):
        """Automatically prune old runs matching DS_*.json in runs_dir."""
        try:
            kept, deleted = self.prune_runs(dry_run=False)
            if deleted:
                logger.info(f"Auto-pruned {len(deleted)} old runs")
        except Exception as e:
            logger.warning(f"Auto-prune failed: {e}")


# Global instance
_run_manager = None


def get_run_manager() -> RunManager:
    """Get the global run manager instance."""
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager()
    return _run_manager