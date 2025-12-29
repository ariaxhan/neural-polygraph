#!/usr/bin/env python3
"""
Universal Experiment Runner for Neural Polygraph

Mirrors the experiments repo architecture for running experiments with
robust storage, clean imports, and easy execution.

Usage:
    python run_experiment.py 01_spectroscopy
    python run_experiment.py 02_geometry
    python run_experiment.py 03_ghost_features
    
    # List available experiments
    python run_experiment.py --list
    
    # View results
    python run_experiment.py --view 01_spectroscopy
"""

import sys
import argparse
from pathlib import Path
from importlib import import_module

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


EXPERIMENTS = {
    "01_spectroscopy": {
        "name": "Pure Spectroscopy",
        "description": "Spectral signatures of hallucinations (L0, L2, Gini)",
        "module": "experiments.01_spectroscopy",
        "status": "ready",
    },
    "02_geometry": {
        "name": "Geometric Topology",
        "description": "Inertia tensor analysis of feature distributions",
        "module": "experiments.02_geometry",
        "status": "coming_soon",
    },
    "03_ghost_features": {
        "name": "Ghost Features",
        "description": "Differential spectrum analysis",
        "module": "experiments.03_ghost_features",
        "status": "coming_soon",
    },
}


def list_experiments():
    """List all available experiments."""
    print("=" * 80)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 80)
    print()
    
    for exp_id, info in EXPERIMENTS.items():
        status_icon = "✅" if info["status"] == "ready" else "🚧"
        print(f"{status_icon} {exp_id}: {info['name']}")
        print(f"   {info['description']}")
        print(f"   Status: {info['status']}")
        print()
    
    print("Usage:")
    print("  python run_experiment.py 01_spectroscopy")
    print("  python run_experiment.py --view 01_spectroscopy")
    print()


def view_results(experiment_id: str):
    """View results for an experiment."""
    from hallucination_detector import ExperimentStorage
    
    experiment_path = Path("experiments") / experiment_id
    
    if not experiment_path.exists():
        print(f"Error: Experiment '{experiment_id}' not found")
        return
    
    try:
        # Check for existing runs first to avoid creating a new run directory
        runs_path = experiment_path / "runs"
        latest_run = None
        
        if runs_path.exists():
            runs = [
                d.name for d in runs_path.iterdir()
                if d.is_dir() and (d / "metrics.parquet").exists()
            ]
            runs = sorted(runs, reverse=True)
            latest_run = runs[0] if runs else None
        
        # Initialize storage with latest run if it exists, otherwise it will create a new one
        storage = ExperimentStorage(experiment_path, run_id=latest_run) if latest_run else ExperimentStorage(experiment_path)
        summary = storage.get_summary()
        
        print("=" * 80)
        print(f"EXPERIMENT: {experiment_id}")
        print("=" * 80)
        print()
        print(f"Experiment Path: {summary['experiment_path']}")
        print(f"Total Runs: {summary['total_runs']}")
        print(f"Latest Run: {summary['latest_run']}")
        print()
        
        if summary['total_runs'] > 0:
            print("All Runs:")
            for run_id in summary['all_runs']:
                size = summary.get('run_sizes', {}).get(run_id, 'unknown')
                print(f"  - {run_id} ({size} rows)")
            print()
            
            # Try to load and show summary stats from latest run
            try:
                import polars as pl
                # Use latest_run from summary (which we already set as storage.run_id)
                df = storage.read_metrics()
                print("Latest Run Summary:")
                print(f"  Total samples: {len(df)}")
                if 'domain' in df.columns:
                    print(f"  Domains: {df['domain'].unique().to_list()}")
                if 'condition' in df.columns:
                    print(f"  Conditions: {df['condition'].unique().to_list()}")
                print()
            except Exception as e:
                print(f"  Could not load metrics: {e}")
                print()
        else:
            print("No runs found. Run the experiment first:")
            print(f"  python run_experiment.py {experiment_id}")
            print()
    
    except Exception as e:
        print(f"Error viewing results: {e}")


def export_results(experiment_id: str, output_file: str, run_id: str = None, all_runs: bool = False):
    """Export experiment results to CSV."""
    from hallucination_detector import ExperimentStorage
    
    experiment_path = Path("experiments") / experiment_id
    
    if not experiment_path.exists():
        print(f"Error: Experiment '{experiment_id}' not found")
        return
    
    try:
        # Check for existing runs first
        runs_path = experiment_path / "runs"
        latest_run = None
        
        if runs_path.exists():
            runs = [
                d.name for d in runs_path.iterdir()
                if d.is_dir() and (d / "metrics.parquet").exists()
            ]
            runs = sorted(runs, reverse=True)
            latest_run = runs[0] if runs else None
        
        # Use specified run_id, or latest_run, or create new
        if run_id:
            storage = ExperimentStorage(experiment_path, run_id=run_id)
        elif latest_run:
            storage = ExperimentStorage(experiment_path, run_id=latest_run)
        else:
            storage = ExperimentStorage(experiment_path)
        
        # Load data
        if all_runs:
            df = storage.read_all_runs()
            print(f"Loading all runs combined...")
        else:
            df = storage.read_metrics()
            print(f"Loading run: {storage.run_id}")
        
        # Export
        df.write_csv(output_file)
        print(f"\n✓ Exported {len(df)} rows to {output_file}")
        print(f"  File: {Path(output_file).absolute()}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nAvailable runs:")
        try:
            runs = storage.list_runs()
            if runs:
                for r in runs:
                    print(f"  - {r}")
            else:
                print("  No runs found. Run the experiment first:")
                print(f"    python run_experiment.py {experiment_id}")
        except:
            pass
    except Exception as e:
        print(f"\n✗ Error exporting: {e}")
        import traceback
        traceback.print_exc()


def run_experiment(experiment_id: str):
    """Run a specific experiment."""
    if experiment_id not in EXPERIMENTS:
        print(f"Error: Unknown experiment '{experiment_id}'")
        print()
        list_experiments()
        return
    
    exp_info = EXPERIMENTS[experiment_id]
    
    if exp_info["status"] != "ready":
        print(f"Error: Experiment '{experiment_id}' is not ready yet")
        print(f"Status: {exp_info['status']}")
        return
    
    print("=" * 80)
    print(f"RUNNING: {exp_info['name']}")
    print("=" * 80)
    print(f"Description: {exp_info['description']}")
    print()
    
    # Import and run the experiment module
    try:
        # Import the experiment script
        module_path = Path("experiments") / f"{experiment_id}.py"
        
        if not module_path.exists():
            print(f"Error: Experiment script not found: {module_path}")
            return
        
        # Execute the experiment script
        print(f"Executing: {module_path}")
        print()
        
        # Run the script
        import subprocess
        result = subprocess.run(
            [sys.executable, str(module_path)],
            cwd=Path.cwd(),
        )
        
        if result.returncode == 0:
            print()
            print("=" * 80)
            print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print()
            print("View results:")
            print(f"  python run_experiment.py --view {experiment_id}")
            print()
        else:
            print()
            print("=" * 80)
            print("❌ EXPERIMENT FAILED")
            print("=" * 80)
            print()
    
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural Polygraph Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py 01_spectroscopy              # Run spectroscopy experiment
  python run_experiment.py --list                       # List all experiments
  python run_experiment.py --view 01_spectroscopy       # View results
  python run_experiment.py --export 01_spectroscopy    # Export to results.csv
  python run_experiment.py --export 01_spectroscopy --output my_results.csv
  python run_experiment.py --export 01_spectroscopy --all-runs  # Export all runs
        """
    )
    
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment ID to run (e.g., 01_spectroscopy)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments"
    )
    parser.add_argument(
        "--view",
        metavar="EXPERIMENT_ID",
        help="View results for an experiment"
    )
    parser.add_argument(
        "--export",
        metavar="EXPERIMENT_ID",
        help="Export results for an experiment"
    )
    parser.add_argument(
        "--output",
        metavar="FILENAME",
        default="results.csv",
        help="Output filename for export (default: results.csv)"
    )
    parser.add_argument(
        "--run",
        metavar="RUN_ID",
        help="Specific run ID to export (default: latest)"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Export all runs combined"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.view:
        view_results(args.view)
    elif args.export:
        export_results(args.export, args.output, args.run, args.all_runs)
    elif args.experiment:
        run_experiment(args.experiment)
    else:
        parser.print_help()
        print()
        list_experiments()


if __name__ == "__main__":
    main()




