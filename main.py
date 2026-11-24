"""
DataSmith AI - Main Entry Point
Simple CLI for users to input their dataset requirements
"""

from agents.requirement_analyzer import RequirementAnalyzer
from core.dataset_spec import create_initial_spec, PipelineStatus
from core.llm_wrapper import get_llm
from core.run_manager import get_run_manager
import json
import logging
import os
import sys
import time
from pathlib import Path

def setup_logging(log_dir=None):
    """Write all log messages to logs/datasmith.log.

    Set the environment variable DATASMITH_DEBUG=1 for more detail.
    Returns the file handler so tests can remove it again.
    """
    log_dir = Path(log_dir) if log_dir else Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_dir / "datasmith.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if os.environ.get("DATASMITH_DEBUG") else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # silence per-request noise
    return handler


def warm_up_ollama():
    """Warm up Ollama by sending a tiny request."""
    print("\n⏳ Warming up Ollama (loading model into memory)...")
    start = time.time()
    
    try:
        llm = get_llm()
        # Tiny warm-up call - just ping with JSON response
        llm.call(
            prompt='Respond with exactly: {"status": "ok"}',
            expected_schema=None,
            force_json=True
        )
        duration = time.time() - start
        print(f"✅ Ollama warmed up in {duration:.2f}s (model loaded)\n")
        return duration
    except Exception as e:
        duration = time.time() - start
        # Warn but continue - warm-up is optional
        print(f"⚠️  Warm-up took {duration:.2f}s but had issues: {str(e)[:100]}")
        print("   Continuing anyway...\n")
        return duration


def main():
    """Main CLI interface for DataSmith AI."""
    setup_logging()
    # Check for --clean command
    if len(sys.argv) > 1 and sys.argv[1] == '--clean':
        force_clean = '--force' in sys.argv or '-f' in sys.argv
        clean_runs(force=force_clean)
        return
    
    print("\n" + "="*70)
    print("Welcome to DataSmith AI - Dataset Requirement Analyzer")
    print("="*70)
    print("\nThis tool helps you find and prepare environmental/nature datasets.")
    print("Focus: Weather, air quality, earthquakes, fires, ocean, satellites, etc.\n")
    
    # Auto-prune old runs at startup
    run_manager = get_run_manager()
    
    # Warm up Ollama
    warm_up_ollama()
    
    # Get dataset name
    dataset_name = input("Enter a name for your dataset: ").strip()
    if not dataset_name:
        dataset_name = "My Dataset"
    
    # Get description
    print("\nDescribe what data you need (be as specific as possible):")
    print("Example: 'live air quality data for Delhi, last 7 days, CSV'")
    description = input("\nYour requirement: ").strip()
    
    if not description:
        print("\n❌ Error: Description cannot be empty!")
        return
    
    # Analyze using Orchestrator
    print("\n" + "="*70)
    print("Analyzing your requirement...")
    print("="*70)
    
    from core.orchestrator import Orchestrator, AgentDefinition
    
    start_time = time.time()
    orchestrator = Orchestrator(auto_prune=True)
    analyzer = RequirementAnalyzer(interactive=True)  # Interactive mode for main.py
    
    agent_def = AgentDefinition(
        name="RequirementAnalyzer",
        run_func=analyzer.analyze,
        required_inputs=[],
        outputs=["requirement"],
        skip_if_unsupported=False
    )
    orchestrator.register_agent(agent_def)
    
    result = orchestrator.run(dataset_name, description)
    analysis_time = time.time() - start_time

    if result.pipeline_status == PipelineStatus.PAUSED:
        print("\n" + "="*70)
        print("CLARIFICATION REQUIRED")
        print("="*70)
        if result.requirement and result.requirement.clarifying_questions:
            for i, question in enumerate(result.requirement.clarifying_questions, 1):
                print(f"   {i}. {question}")
        print("\nPlease re-run with a clearer description.")
        run_manager.save_spec(result)
        return
    
    # Display results
    req = result.requirement
    
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    if not req.is_supported:
        print(f"\n❌ Request Not Supported")
        print(f"   Reason: {req.unsupported_reason}")
        return
    
    print(f"\n📊 Dataset: {req.dataset_name}")
    print(f"   Description: {req.description}")
    
    print(f"\n🔍 Classification:")
    print(f"   Domain: {req.domain}")
    print(f"   Subdomain: {req.subdomain if req.subdomain else 'Not specified'}")
    print(f"   Problem Type: {req.problem_type}")
    print(f"   Data Modality: {req.data_modality.value if req.data_modality else 'Not specified'}")    
    print(f"\n📍 Details:")
    if req.geography:
        print(f"   Geography: {req.geography}")
    if req.time_range:
        print(f"   Time Range: {req.time_range}")
    if req.time_granularity:
        print(f"   Time Granularity: {req.time_granularity}")
    if req.freshness_need != "unspecified":
        print(f"   Freshness: {req.freshness_need}")
    if req.max_data_age:
        print(f"   Max Data Age: {req.max_data_age}")
    if req.expected_size:
        print(f"   Expected Size: {req.expected_size}")
    print(f"   Output Format: {req.output_format}")
    
    if req.expected_features:
        print(f"\n📋 Expected Input Features: {', '.join(req.expected_features)}")
    
    if req.target_variable:
        print(f"   Target Variable: {req.target_variable}")
    if req.constraints:
        print(f"\n⚠️  Constraints: {', '.join(req.constraints)}")
    
    print(f"\n✅ Completeness Score: {req.completeness:.2f} / 1.00")
    print(f"   ({round(req.completeness * 100)}% of details provided)")

    
    if req.explicitly_stated:
        print(f"\n✓ What you specified:")
        for item in req.explicitly_stated:
            print(f"   • {item}")
    
    if req.missing_or_unclear:
        print(f"\n❓ Missing or unclear information:")
        for item in req.missing_or_unclear:
            print(f"   • {item}")
    
    if req.clarifying_questions:
        print(f"\n💬 Clarifying questions:")
        for i, question in enumerate(req.clarifying_questions, 1):
            print(f"   {i}. {question}")
    
    # Save to file using RunManager
    run_manager = get_run_manager()
    output_path = run_manager.save_spec(result)
    
    print(f"\n💾 Full results saved to: {output_path}")
    print(f"⏱️  Analysis time: {analysis_time:.2f}s")
    print("\n" + "="*70)
    print("Thank you for using DataSmith AI!")
    print("="*70 + "\n")


def clean_runs(force: bool = False):
    """Clean old runs. Dry-run by default unless force=True."""
    print("\n" + "="*70)
    print("DataSmith AI - Run Cleanup")
    print("="*70)
    
    run_manager = get_run_manager()
    
    print("\n🔍 Analyzing runs to delete (dry run)...")
    kept, to_delete = run_manager.prune_runs(dry_run=True)
    
    if not to_delete:
        print("✨ No old runs to delete - everything is current!")
        print(f"\nKept {len(kept)} files:")
        for filename in kept:
            print(f"   • {filename}")
    else:
        print(f"\n📋 Cleanup plan (keep_last=3, resumable_max_age=7 days):")
        print(f"\n✅ Keeping {len(kept)} files:")
        for filename in kept:
            print(f"   • {filename}")
        
        print(f"\n🗑️  Would delete {len(to_delete)} old files:")
        for filename in to_delete:
            print(f"   • {filename}")
        
        if force:
            print("\n⚡ --force specified: proceeding with deletion...")
            kept_actual, deleted_actual = run_manager.prune_runs(dry_run=False)
            print(f"✅ Deleted {len(deleted_actual)} files")
            print(f"💾 Kept {len(kept_actual)} files")
        else:
            try:
                response = input(f"\nDelete {len(to_delete)} old run files? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                response = 'n'
            
            if response == 'y':
                kept_actual, deleted_actual = run_manager.prune_runs(dry_run=False)
                print(f"\n✅ Deleted {len(deleted_actual)} files")
                print(f"💾 Kept {len(kept_actual)} files")
            else:
                print("\n❌ Cancelled - no files deleted")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
