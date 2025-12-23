# Neural Polygraph Setup Guide

This guide will help you get the repository up and running for your Medium article series.

## ✅ Repository Structure Created

The neural-polygraph repository has been initialized with the following structure:

```
neural-polygraph/
├── README.md                          # Main repository documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Package configuration and dependencies
├── SETUP_GUIDE.md                     # This file
│
├── tutorials/                         # Educational notebooks
│   ├── README.md                      # Tutorial guide
│   ├── 01_sae_basics.ipynb           # Introduction to SAEs
│   └── 02_feature_extraction.ipynb   # Feature comparison
│
├── experiments/                       # Research experiments
│   ├── README.md                      # Experiment documentation
│   ├── hallucination_biopsy.py       # Main experiment script
│   ├── results/                      # Output directory
│   └── notebooks/                    # Analysis notebooks
│
└── src/hallucination_detector/       # Reusable package
    ├── __init__.py                    # Package exports
    └── sae_utils.py                  # Core functions
```

## 🚀 Next Steps

### 1. Initialize Git Repository

```bash
cd /Users/ariahan/Documents/ai-research/neural-polygraph
git init
git add .
git commit -m "Initial repository structure for hallucination detection"
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode (includes all dependencies)
pip install -e .

# Or install with dev dependencies for development
pip install -e ".[dev]"
```

This will install all dependencies from `pyproject.toml`:
- `torch` - PyTorch for tensor operations
- `transformer-lens` - For model activation access
- `sae-lens` - For SAE loading and encoding
- `numpy` - Numerical operations
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `scikit-learn`, `umap-learn` - Machine learning tools
- `jupyter`, `ipywidgets`, `notebook`, `jupyterlab` - Notebook interface

### 4. Test the Installation

```bash
# Test imports
python -c "from hallucination_detector import initialize_model_and_sae; print('✓ Package installed correctly')"
```

### 5. Run the Experiment

```bash
python experiments/hallucination_biopsy.py
```

**Note:** First run will download ~5GB of models. Subsequent runs use cached versions.

### 6. Launch Jupyter Notebooks

```bash
jupyter notebook tutorials/
```

Start with `01_sae_basics.ipynb` and work through sequentially.

## 📝 Code Organization

### Core Functions (src/hallucination_detector/sae_utils.py)

The package provides these key functions:

1. **`initialize_model_and_sae(device=None)`**
   - Loads Gemma-2-2b and GemmaScope SAE
   - Auto-detects device (MPS/CUDA/CPU)
   - Returns: (model, sae, device)

2. **`extract_features(text, model, sae)`**
   - Extracts SAE feature activations from text
   - Returns: dict with indices, magnitudes, total_active, energy

3. **`decode_feature(feature_id, model, sae, top_k=5)`**
   - Translates feature to vocabulary words
   - Returns: dict with feature_id, words, logits

4. **`get_loudest_unique_features(fact_text, hall_text, model, sae, top_k=5)`**
   - Finds features unique to hallucination
   - Sorted by activation magnitude
   - Returns: list of feature indices

5. **`run_differential_diagnosis(fact_text, hall_text, model, sae)`**
   - Complete comparative analysis
   - Returns: dict with spectral_metrics, biomarkers, signatures

### Experiment Script (experiments/hallucination_biopsy.py)

Demonstrates the full methodology:
- Loads instruments
- Defines test case (fact vs hallucination)
- Runs differential diagnosis
- Identifies loudest unique features
- Decodes feature meanings
- Saves results to JSON

### Tutorial Notebooks

**01_sae_basics.ipynb:**
- Introduction to SAEs
- The "prism metaphor"
- Manual feature extraction
- Feature decoding

**02_feature_extraction.ipynb:**
- Using the hallucination_detector package
- Comparing multiple texts
- Finding unique features
- Differential diagnosis

## 🎯 For Your Medium Articles

### Article 1: "The Prism Metaphor"
- Use: `tutorials/01_sae_basics.ipynb`
- Focus: What are SAEs, why they matter
- Code: Simple feature extraction examples
- Visuals: Feature activation patterns

### Article 2: "Spectral Signatures"
- Use: `tutorials/02_feature_extraction.ipynb`
- Focus: Comparing fact vs hallucination
- Code: Differential diagnosis methodology
- Visuals: Unique feature comparisons

### Article 3: "Hallucination Biomarkers"
- Use: `experiments/hallucination_biopsy.py`
- Focus: Experimental findings
- Code: Full experiment with results
- Visuals: Feature translations, energy differences

## 📊 Expected Results

Based on your experiments repo results:

```
Geography Teleportation:
  Unique features: 73
  Energy diff: +116.143
  Top feature: #9958 → RB, RSD, RCS

Geography Teleportation 2:
  Unique features: 40
  Energy diff: -136.787
  Top feature: #10496 → York, YORK, York
```

The simplified experiment in this repo uses a single example for clarity and speed.

## 🔧 Customization

### Adding More Test Cases

Edit `experiments/hallucination_biopsy.py`:

```python
# Add more test cases in main()
test_cases = [
    {
        "fact": "Your factual statement",
        "hallucination": "Your hallucinated version"
    },
    # Add more...
]
```

### Using Different Models/SAEs

Edit `src/hallucination_detector/sae_utils.py`:

```python
# In initialize_model_and_sae()
model_name = "gemma-2-2b"  # Change model
sae_release = "gemma-scope-2b-pt-res-canonical"  # Change SAE
sae_id = "layer_5/width_16k/canonical"  # Change layer/width
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the venv
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Model Download Issues
```bash
# Check Hugging Face access
huggingface-cli whoami

# Login if needed
huggingface-cli login
```

### Memory Issues
- Close other applications
- Use CPU instead of GPU (automatic fallback)
- Reduce batch size (already minimal)

## 📚 Resources

- **SAE Lens:** https://github.com/jbloomAus/SAELens
- **TransformerLens:** https://github.com/neelnanda-io/TransformerLens
- **Neuronpedia:** https://neuronpedia.org/gemma-2b
- **GemmaScope:** https://huggingface.co/google/gemma-scope

## ✨ Ready to Publish

The repository is now ready for:
- ✅ GitHub publication
- ✅ Medium article series
- ✅ Community experimentation
- ✅ Further research

## 📧 Next Actions

1. **Test the setup:**
   ```bash
   python experiments/hallucination_biopsy.py
   jupyter notebook tutorials/01_sae_basics.ipynb
   ```

2. **Create GitHub repo:**
   - Create new repo on GitHub
   - Add remote: `git remote add origin <url>`
   - Push: `git push -u origin main`

3. **Write Medium articles:**
   - Use notebooks as interactive examples
   - Reference experiment results
   - Link to GitHub repo

4. **Share with community:**
   - Post on Twitter/X
   - Share in ML/AI communities
   - Engage with feedback

---

**Good luck with your Medium article series!** 🚀

