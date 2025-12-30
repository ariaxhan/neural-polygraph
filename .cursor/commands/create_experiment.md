# Create Experiment Command

●command|create_experiment|streamlined:true|parallel:optimized

## Purpose
Create a new parallelized experiment file with minimal overhead. Just one file, clean and optimized.

## Usage
```
create experiment [type]
```

Example:
```
create experiment feature_dynamics
create experiment layer_analysis
create experiment temporal_patterns
```

## What Gets Created
✓ **ONE file**: `experiments/number_[type].py` (parallelized, optimized)
count number sequentially, look at the last file in the experiments directory and use the next number
✓ **README update**: Add entry to experiments/README.md
✗ No guides, no summaries, no extra markdown files

## Behavior
→generate|single_file:experiments/0X_[type].py|parallel:true|optimized:true
→update|experiments/README.md|add_entry:true
→no_extra_files|no_guides:true|no_summaries:true|streamlined:true

## Template
→base|06_comprehensive_analysis_parallel.py|proven:optimized
→structure|batch_processor:class|modules:4_5|timing:enabled
→optimization|batch_size:16|parallel:n_jobs_minus_1|vectorized:true

## File Naming
→pattern|0X_[type].py|X:next_number|type:user_provided
→examples|08_feature_dynamics.py|09_layer_analysis.py|10_temporal_patterns.py

## What Gets Created
✓ Single experiment file (0X_[type].py)
✓ Updated experiments/README.md
✗ No separate guides
✗ No summary files
✗ No workflow docs
✗ No checklists

## Template Structure
```python
#!/usr/bin/env python3
"""
Experiment 0X: [Type] (OPTIMIZED)

PARALLELIZATION STRATEGIES:
1. Batch Processing: [...]
2. Vectorized Operations: [...]
3. Parallel Computation: [...]

Expected Speedup: 3-5x faster than sequential
"""

import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hallucination_detector import (
    HB_Benchmark,
    ExperimentStorage,
)


@dataclass
class BatchSample:
    """Container for batch processing."""
    domain: str
    sample: object
    # Add fields as needed


class BatchProcessor:
    """Optimized batch processor."""
    
    def __init__(self, benchmark: HB_Benchmark, device: str, batch_size: int = 16):
        self.benchmark = benchmark
        self.device = device
        self.batch_size = batch_size
    
    def batch_get_activations(self, texts: List[str]) -> List[Dict]:
        """Get activations for multiple texts in a single forward pass."""
        # Batch tokenization
        # Single forward pass
        # Batch SAE encoding
        pass
    
    def process_batch(self, batch: List[BatchSample]) -> List[Dict]:
        """Process a batch of samples."""
        pass


def run_experiment():
    """Main experiment runner (OPTIMIZED)."""
    start_time = time.time()
    
    print("=" * 80)
    print("EXPERIMENT 0X: [TYPE] (OPTIMIZED)")
    print("=" * 80)
    print()
    
    # Initialize storage
    experiment_path = Path(__file__).parent / "0X_[type]"
    storage = ExperimentStorage(experiment_path)
    
    # Load benchmark
    print("Loading benchmark...")
    benchmark = HB_Benchmark(data_dir="experiments/data")
    benchmark.load_datasets(domains=["entity", "temporal", "logical", "adversarial"])
    benchmark.load_model_and_sae(layer=5, width="16k")
    
    all_samples = benchmark.get_all_samples()
    print(f"Total samples: {len(all_samples)}")
    print()
    
    # Process with batch optimization
    processor = BatchProcessor(benchmark, str(benchmark.model.cfg.device), batch_size=16)
    
    # [Your experiment logic here]
    
    # Save results
    manifest = {
        "experiment_type": "[type]",
        "experiment_name": "0X_[type]",
        "timestamp": datetime.now().isoformat(),
    }
    storage.write_manifest(manifest)
    
    total_time = time.time() - start_time
    print(f"Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return storage


if __name__ == "__main__":
    storage = run_experiment()
```

## README Update Format
```markdown
### Experiment 0X: [Type]
**Status**: ✓ Complete  
**Runtime**: ~X minutes  
**Optimization**: Parallelized (3-5x speedup)  
**Command**: `python experiments/0X_[type].py`

Brief description of what this experiment does.
```

○reference|patterns:PT:103|optimization:batch_parallel|streamlined:minimal_docs

