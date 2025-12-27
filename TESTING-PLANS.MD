●STATE|Ψ:lab_architect|Ω:targeted_experiments
PAYLOAD :: RESEARCH LAB GUIDE
"The Shape of Errors: Spectroscopic and Geometric Analysis of SAE Features"
Scope: SAEs as measurement tools for latent space topology during hallucinations.
Model: Gemma-2-2B (via GemmaScope)
Target Output: 4 High-Impact Figures + Quantitative Metrics

### PHASE 1: PREPARATION & INFRASTRUCTURE

**1.1 The "HB-1000" Benchmark Suite (Hallucination Benchmark)**
We move beyond small samples to a robust, domain-segmented benchmark to prove the concept across different types of reasoning.

* **Structure:** ~1,000 Triples `(Prompt, Fact_Completion, Hallucination_Completion)` stored in 4 targeted JSON files.
* **Datasets:**
* `bench_entity_swaps.json`: Concrete nouns/locations (Paris  Rome).
* `bench_temporal_shifts.json`: Continuous variable errors (1969  1975).
* `bench_logical_inversions.json`: Structural/Relational flips (A > B  B > A).
* `bench_adversarial_traps.json`: High-probability misconceptions designed to fool detectors.


* **Control:** Length-matched to prevent "length correlates with sparsity" artifacts.

**1.2 The Measurement Rig (GemmaScope)**

* **Target:** Layer 12 and Layer 20 (Mid-level semantics vs. Output logic).
* **Extraction:** Residual stream activations  SAE Encoder  Feature Activations.

---

### PHASE 2: EXPERIMENTAL OUTLINE

#### Experiment A: Pure Spectroscopy (The "Prism" Effect)

*Goal: Demonstrate that hallucinations have a distinct spectral signature (entropy/sparsity).*

* **Metric 1:  Norm (Sparsity).** Do hallucinations trigger *more* features (confusion) or *fewer* (mode collapse)?
* **Metric 2: Reconstruction Error ().** Does the SAE fail to reconstruct the hallucination, indicating it lies "off-manifold"?
* **Metric 3: The "Ghost" Features.** Identify features that strictly fire for the hallucination but not the fact.
* *Hypothesis:* Hallucinations activate "loose association" features that lack inhibition.



#### Experiment B: Geometric Topology (The "AIDA-TNG" Angle)

*Goal: Treat active features as point masses in high-dimensional space and measure the "shape" of the thought.*

* **Method:** Calculate the **Inertia Tensor**  of the active features weighted by activation strength.
* 
 where  is activation magnitude and  is the feature vector.


* **Metric 1: Principal Axis Ratios.** Compute eigenvalues .
* Calculated Sphericity ( and ).
* *Hypothesis:* Facts are **Triaxial** (highly structured, specific direction). Hallucinations are **Spherical** (isotropic confusion) or **Oblate** (flat, lacking depth).


* **Metric 2: Substructure/Clumping.** Do active features form one tight cluster (coherent thought) or multiple disconnected islands (schizophrenic representation)?

#### Experiment C: Complexity Scaling

*Goal: Show the method validates Anthropic’s "superposition" claims.*

* Compare "Simple Facts" (Entity Swaps) vs. "Complex Reasoning" (Logical Inversions).
* *Hypothesis:* Complex facts utilize higher-dimensional geometric structures; hallucinations collapse these structures into lower dimensions.

---

### PHASE 3: CURSOR "LAB GUIDE" PROMPTS

*Copy these blocks directly into Cursor to generate the experimental code.*

#### PROMPT 1: Data & Infrastructure

```markdown
●TASK: implement_data_loader
●CONTEXT: research_paper_prep
Create a python script `src/data_loader.py` that:
1. Defines the `HB_Benchmark` class.
2. Implements a loader that reads from the 4 specific dataset files:
   - `data/bench_entity_swaps.json`
   - `data/bench_temporal_shifts.json`
   - `data/bench_logical_inversions.json`
   - `data/bench_adversarial_traps.json`
3. Implements a method to load Gemma-2-2B and the corresponding GemmaScope SAEs (using transformer_lens and sae_lens).
4. Creates a `get_activations(text)` function that returns:
   - Original Model Residual Stream
   - SAE Feature Activations (indices and magnitudes)
   - SAE Reconstruction Error
Ensure strictly typed Python.

```

#### PROMPT 2: Experiment A (Spectroscopy)

```markdown
●TASK: run_spectroscopy
●CONTEXT: feature_analysis
Create `experiments/01_spectroscopy.py`.
Using the data loader:
1. Iterate through all 4 datasets (Entity, Temporal, Logical, Adversarial).
2. For both FACT and HALLUCINATION, compute:
   - L0 Norm (number of active features > 0.0).
   - L2 Norm of the Reconstruction Error vector.
   - Gini Coefficient of the activation magnitudes (measure of "focus").
3. Store results in a Pandas DataFrame, including the 'domain' column (e.g., 'entity', 'logical').
4. Visualization: Create a comparative Histogram (Fact vs Hallucination) for Reconstruction Error, faceted by Domain.
   - Hypothesis: Hallucinations have higher reconstruction error (off-manifold), especially in logical tasks.

```

#### PROMPT 3: Experiment B (Geometry/Topology)

```markdown
●TASK: run_geometric_analysis
●CONTEXT: aida_tng_topology
Create `experiments/02_geometry.py`.
This is the core novelty. We treat feature activations as physical distributions.
1. Import the `compute_inertia_tensor` function from `src/geometry.py`.
2. Iterate through the datasets. For a given input's active features (indices $i$, magnitudes $w$):
   - Retrieve the decoder weights $W_{dec}$ for the active features.
   - Run `compute_inertia_tensor` to get shape metrics (Sphericity, Misalignment).
3. Output a scatter plot: X-axis = Sphericity, Y-axis = L0 Norm.
   - Color code points by "Fact" (Blue) vs "Hallucination" (Red).

```

#### PROMPT 4: The "Ghost Feature" Finder

```markdown
●TASK: find_divergent_features
●CONTEXT: interpretability
Create `experiments/03_ghost_features.py`.
1. Calculate the "Differential Spectrum": $Act_{diff} = Act_{hallucination} - Act_{fact}$.
2. Identify the "Top-K" features with the highest positive value in $Act_{diff}$ (features unique to the lie).
3. Use the SAE Lens functionality to pull the auto-interpretability dashboard descriptions for these specific features.
4. Output a text report: "For the prompt 'Eiffel Tower', the hallucination 'Rome' activated Feature #1402 (Description: 'Italian cities/architecture') which was silent in the Fact."

```

---

### EXPECTED FIGURES FOR PAPER

| Figure ID | Type | Description | Value Prop |
| --- | --- | --- | --- |
| **Fig 1** | **The Spectral Shift** | Dual-histogram of SAE Reconstruction Error. | Shows hallucinations are "off-manifold" and harder for SAEs to compress. |
| **Fig 2** | **Topological Phase Plot** | Scatter Plot: **Sphericity** () vs **Sparsity** (). Facts should cluster in the "Structured/Sparse" region; Hallucinations in the "Isotropic/Diffuse" region. | **Main Novelty.** Visually proves geometric degradation during hallucinations. |
| **Fig 3** | **Feature Prism** | Case study of one prompt. Left: Fact features (semantic categories aligned). Right: Hallucination features (conflicting semantic categories). | Qualitative proof of "Ghost Features." |
| **Fig 4** | **Layer Sensitivity** | Line chart: Detection accuracy (geometry vs spectroscopy) across layers 10-25. | Shows *where* in the model the thought becomes "untrue." |