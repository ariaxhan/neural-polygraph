#!/usr/bin/env python3
"""Quick verification that everything is set up correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify():
    """Verify imports and data files."""
    print("Verifying Neural Polygraph setup...")
    print()
    
    # Test imports
    try:
        from hallucination_detector import (
            HB_Benchmark,
            ExperimentStorage,
            compute_inertia_tensor,
        )
        print("✅ Core imports working")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data files
    data_dir = Path("experiments/data")
    required_files = [
        "bench_entity_swaps.json",
        "bench_temporal_shifts.json",
        "bench_logical_inversions.json",
        "bench_adversarial_traps.json",
    ]
    
    all_exist = True
    for filename in required_files:
        if (data_dir / filename).exists():
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} missing")
            all_exist = False
    
    print()
    if all_exist:
        print("✅ Setup verified! Ready to run experiments.")
        print()
        print("Next steps:")
        print("  python run_experiment.py --list")
        print("  python run_experiment.py 01_spectroscopy")
        return True
    else:
        print("❌ Setup incomplete. Check missing files.")
        return False

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

