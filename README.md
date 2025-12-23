# Hallucination Detector: SAE Spectral Signatures

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Detecting hallucinations in language models using Sparse Autoencoder (SAE) feature analysis.**

This repository demonstrates a novel approach to identifying hallucinations by analyzing the "spectral signatures" of model activations. By comparing feature patterns between factual and hallucinated text, we can identify unique biomarkers that indicate when a model is generating false information.

## Quick Start

```python
from hallucination_detector import (
    initialize_model_and_sae,
    get_loudest_unique_features,
    decode_feature
)

# Load model and SAE
model, sae, device = initialize_model_and_sae()

# Compare fact vs hallucination
fact = "The Eiffel Tower is in Paris"
hallucination = "The Eiffel Tower is in Rome"

# Find unique features
unique_features = get_loudest_unique_features(fact, hallucination, model, sae)

# Decode what they mean
for feat_id in unique_features:
    decoded = decode_feature(feat_id, model, sae)
    print(f"Feature #{feat_id} → {decoded['words']}")
```

## Repository Guide

### 🎓 **New to SAEs?** Start here:
1. Read `tutorials/01_sae_basics.ipynb` - Learn what SAEs are and how they work
2. Read `tutorials/02_feature_extraction.ipynb` - Learn to compare features between texts
3. Run `experiments/hallucination_biopsy.py` - See the full methodology in action

### 🔬 **Want to see the research?**
- Run `experiments/hallucination_biopsy.py` for the core experiment
- Check `experiments/results/` for saved outputs
- Read the Medium article series (links below)

### 🛠️ **Want to use the code?**
- Install the package: `pip install -e .`
- Import functions from `hallucination_detector`
- See Quick Start above for usage examples

## Installation

### Requirements
- Python 3.10+
- ~5GB disk space for model downloads
- Apple Silicon (MPS), CUDA GPU, or CPU

### Setup

```bash
# Clone the repository
git clone https://github.com/ariaxhan/neural-polygraph.git
cd neural-polygraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode (includes all dependencies)
pip install -e .

# Run the experiment
python experiments/hallucination_biopsy.py
```

## Key Findings

Our analysis of Gemma-2-2b using GemmaScope SAEs reveals:

- **Unique Signatures:** Hallucinations activate 40-100+ features not present in factual text
- **Energy Differences:** Hallucinations show distinct energy patterns (±50-500 units)
- **Interpretable Biomarkers:** Top features decode to semantically relevant concepts
  - Geography errors → location-specific features
  - Temporal errors → time/era-specific features
  - Biological errors → anatomy/capability features

Example from our experiments:
```
Fact: "The Eiffel Tower is in Paris"
Hallucination: "The Eiffel Tower is in Rome"

Top unique feature: #10496 → "York", "YORK", "York"
(Activates for wrong geographic locations)
```

## Methodology

1. **Load Instruments:** Gemma-2-2b model + GemmaScope SAE (layer 5, 16k features)
2. **Extract Features:** Run text through model, apply SAE to get sparse activations
3. **Compare Signatures:** Identify features unique to hallucination
4. **Decode Biomarkers:** Project features onto vocabulary to interpret meaning
5. **Analyze Patterns:** Look for consistent hallucination signatures

See `tutorials/` for detailed walkthroughs.

## Project Structure

```
neural-polygraph/
├── README.md                          # This file
├── tutorials/                         # Educational notebooks
│   ├── 01_sae_basics.ipynb           # Introduction to SAEs
│   └── 02_feature_extraction.ipynb   # Feature comparison techniques
├── experiments/                       # Research experiments
│   ├── hallucination_biopsy.py       # Main experiment script
│   └── results/                      # Saved experiment outputs
└── src/hallucination_detector/       # Reusable package
    ├── __init__.py
    └── sae_utils.py                  # Core functions
```

## Article Series

This repository accompanies a Medium article series on hallucination detection:

1. **Part 1: Why Prompt Engineering Can't Fix Hallucinations (But Neurosurgery Can)** - Introduction to SAEs and how they can be used to detect hallucinations
2. More to come 

*(Links to be added upon publication)*

## Acknowledgments

- **SAE Lens:** For the excellent SAE library and GemmaScope models
- **TransformerLens:** For easy access to model activations
- **Neuronpedia:** For feature exploration and visualization

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to:
- Try new hallucination types
- Test different models/SAEs
- Improve the methodology
- Add visualizations

Open an issue or PR to get started.
