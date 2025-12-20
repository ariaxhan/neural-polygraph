# Neural Polygraph - Quick Reference Card

## 🚀 Quick Start Commands

```bash
# Setup (one-time)
cd /Users/ariahan/Documents/ai-research/neural-polygraph
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Verify setup
python test_setup.py

# Run experiment
python experiments/hallucination_biopsy.py

# Launch notebooks
jupyter notebook tutorials/
```

---

## 📦 Package API

```python
from hallucination_detector import (
    initialize_model_and_sae,
    extract_features,
    decode_feature,
    get_loudest_unique_features,
    run_differential_diagnosis,
)

# Load model and SAE
model, sae, device = initialize_model_and_sae()

# Extract features from text
features = extract_features("Your text here", model, sae)
# Returns: {'indices': [...], 'magnitudes': [...], 'total_active': N, 'energy': X}

# Decode a feature
decoded = decode_feature(feature_id, model, sae, top_k=5)
# Returns: {'feature_id': N, 'words': [...], 'logits': [...]}

# Find unique features in hallucination
unique = get_loudest_unique_features(fact, hallucination, model, sae, top_k=5)
# Returns: [feat_id1, feat_id2, ...]

# Full comparative analysis
diagnosis = run_differential_diagnosis(fact, hallucination, model, sae)
# Returns: {'spectral_metrics': {...}, 'biomarkers': {...}, 'signatures': {...}}
```

---

## 📁 File Locations

| What | Where |
|------|-------|
| Main README | `README.md` |
| Setup guide | `SETUP_GUIDE.md` |
| Package code | `src/hallucination_detector/sae_utils.py` |
| Experiment script | `experiments/hallucination_biopsy.py` |
| Tutorial 1 | `tutorials/01_sae_basics.ipynb` |
| Tutorial 2 | `tutorials/02_feature_extraction.ipynb` |
| Results | `experiments/results/` |
| Test script | `test_setup.py` |

---

## 🎯 For Medium Articles

### Article 1: "The Prism Metaphor"
- **Notebook:** `tutorials/01_sae_basics.ipynb`
- **Topics:** SAE basics, prism metaphor, feature extraction
- **Code:** Simple examples with "The cat sat on the mat"

### Article 2: "Spectral Signatures"
- **Notebook:** `tutorials/02_feature_extraction.ipynb`
- **Topics:** Feature comparison, unique features, differential diagnosis
- **Code:** Eiffel Tower Paris vs Rome example

### Article 3: "Hallucination Biomarkers"
- **Script:** `experiments/hallucination_biopsy.py`
- **Topics:** Experimental findings, patterns, applications
- **Data:** Reference your experiments repo results

---

## 🔧 Common Tasks

### Add a new test case
Edit `experiments/hallucination_biopsy.py`:
```python
fact = "Your factual statement"
hallucination = "Your hallucinated version"
```

### Change model/SAE
Edit `src/hallucination_detector/sae_utils.py`:
```python
model_name = "gemma-2-2b"  # Change here
sae_release = "gemma-scope-2b-pt-res-canonical"
sae_id = "layer_5/width_16k/canonical"
```

### Export notebook to markdown
```bash
jupyter nbconvert --to markdown tutorials/01_sae_basics.ipynb
```

---

## 📊 Expected Results (from your experiments)

```
Geography Teleportation:
  Unique features: 73
  Energy diff: +116.143
  Top feature: #9958 → RB, RSD, RCS

Geography Teleportation 2:
  Unique features: 40
  Energy diff: -136.787
  Top feature: #10496 → York, YORK, York

Historical Anachronism:
  Unique features: 102
  Energy diff: -578.276
  Top feature: #1059 → <bos>, the, '

Biological Impossibility:
  Unique features: 78
  Energy diff: -54.940
  Top feature: #12485 → wings, Wings, wing

Mathematical Inversion:
  Unique features: 22
  Energy diff: -32.006
  Top feature: #14143 → DeleteBehavior, average, dalamnya

Averages:
  Unique features: 63.0
  Energy diff: -137.173
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `pip install -e .` |
| Model download slow | First run only, ~5GB download |
| Out of memory | Close other apps, use CPU |
| Jupyter not found | `pip install jupyter` |
| Package not found | Activate venv: `source venv/bin/activate` |

---

## 🔗 Useful Links

- **Neuronpedia:** https://neuronpedia.org/gemma-2b
- **SAE Lens:** https://github.com/jbloomAus/SAELens
- **TransformerLens:** https://github.com/neelnanda-io/TransformerLens
- **GemmaScope:** https://huggingface.co/google/gemma-scope

---

## 📝 Git Workflow

```bash
# Initialize
git init
git add .
git commit -m "Initial repository structure"

# Create GitHub repo, then:
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main

# Make changes
git add .
git commit -m "Your commit message"
git push
```

---

## ✅ Pre-Publication Checklist

- [ ] Run `python test_setup.py` - all tests pass
- [ ] Run `python experiments/hallucination_biopsy.py` - completes successfully
- [ ] Test both notebooks - execute all cells without errors
- [ ] Update README.md with your GitHub username/URLs
- [ ] Update CITATION with correct info
- [ ] Add `.gitignore` to exclude venv, results
- [ ] Create GitHub repo and push
- [ ] Test clone on fresh machine
- [ ] Write Medium articles
- [ ] Share with community!

---

**Repository Status:** ✅ Ready for publication

