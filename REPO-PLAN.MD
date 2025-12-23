●HANDOFF|target:cursor|id:repo_migration_001|Ψ:structure_executor|Ω:production_ready_repo

●CONTEXT
├─ EXISTING_CODE_LOCATION: /experiments/ (contains hallucination_biopsy.py)
├─ EXISTING_DOCUMENT: Uploaded document index 4 contains full working code
├─ TASK: Migrate to professional repo structure + create educational notebooks
└─ NO_CODE_GENERATION: Extract and reorganize existing code only

●DIRECTIVE|phase:repository_initialization

  ●STRUCTURE|create_directory_tree
    hallucination-spectroscopy/
    ├─ README.md
    ├─ LICENSE (MIT)
    ├─ .gitignore
    ├─ pyproject.toml
    ├─ tutorials/
    │  ├─ README.md
    │  ├─ 01_sae_basics.ipynb
    │  └─ 02_feature_extraction.ipynb
    ├─ experiments/
    │  ├─ README.md
    │  ├─ hallucination_biopsy.py [MIGRATE FROM EXISTING]
    │  ├─ results/
    │  └─ notebooks/
    │     └─ initial_findings.ipynb
    └─ src/
       └─ hallucination_detector/
          ├─ __init__.py
          ├─ sae_utils.py [EXTRACT FROM EXISTING]
          ├─ detector.py
          └─ visualizer.py

●DIRECTIVE|phase:code_migration

  ●EXTRACT|source:document_index_4|operation:function_isolation
    FROM: hallucination_biopsy.py (lines 29-83)
    FUNCTIONS:
      → initialize_instruments()
      → take_biopsy()
      → get_loudest_unique_features()
      → translate_feature()
      → run_differential_diagnosis()
    
    DESTINATION: src/hallucination_detector/sae_utils.py
    MODIFY: Rename initialize_instruments → initialize_model_and_sae
    MODIFY: Rename take_biopsy → extract_features
    MODIFY: Rename translate_feature → decode_feature
    PRESERVE: All docstrings, type hints, logic

  ●COPY|source:document_index_4|operation:preserve_as_is
    FROM: Full hallucination_biopsy.py
    DESTINATION: experiments/hallucination_biopsy.py
    CHANGE: Update imports to use src.hallucination_detector.sae_utils

●DIRECTIVE|phase:notebook_creation

  ●NOTEBOOK|id:01_sae_basics|location:tutorials/01_sae_basics.ipynb
    CELLS:
      [1] MARKDOWN: Title + learning objectives
      [2] CODE: Imports (torch, sae_lens, transformer_lens, matplotlib)
      [3] MARKDOWN: "What is an SAE?" explanation
      [4] CODE: Load pre-trained SAE (use existing initialize_model_and_sae)
      [5] MARKDOWN: "The Prism Metaphor" - visual explanation
      [6] CODE: Process simple text, show feature activations
      [7] MARKDOWN: "Reading the Features" 
      [8] CODE: Decode top 3 features, show what they represent
      [9] MARKDOWN: Key takeaways + next notebook preview
    
    PRINCIPLES:
      → Educational, not research
      → Use gemma-2-2b + canonical SAE (already working)
      → Simple example text: "The cat sat on the mat"
      → Maximum 10-15 min read time
      → Heavy markdown explanation between code cells

  ●NOTEBOOK|id:02_feature_extraction|location:tutorials/02_feature_extraction.ipynb
    CELLS:
      [1] MARKDOWN: Title "Extracting Features from Any Text"
      [2] CODE: Import from src.hallucination_detector.sae_utils
      [3] MARKDOWN: "Feature Extraction Pipeline"
      [4] CODE: Demo extract_features() on 3 different texts
      [5] MARKDOWN: "Comparing Feature Sets"
      [6] CODE: Compare two texts, show unique/shared features
      [7] MARKDOWN: "Decoding Feature Meanings"
      [8] CODE: Use decode_feature() on interesting features
      [9] MARKDOWN: "Next Steps" - preview experiments/

    PRINCIPLES:
      → Builds on notebook 01
      → Introduces comparison logic (foundation for hallucination work)
      → Uses actual functions from src/
      → Still educational, not revealing findings yet

●DIRECTIVE|phase:documentation

  ●FILE|id:main_readme|location:README.md
    SECTIONS:
      → Title + badges (license, python version)
      → One-sentence hook: "Detecting hallucinations via SAE feature patterns"
      → Quick Start code block (3-5 lines, simple usage)
      → Repository Guide (where to start based on background)
      → Installation instructions
      → Article Series section (placeholder links)
      → License footer
    
    TONE: Professional, accessible, confident but not overclaiming
    LENGTH: ~200 lines max
    FORMAT: Standard GitHub markdown

  ●FILE|id:tutorials_readme|location:tutorials/README.md
    CONTENT:
      → Purpose: Educational introduction
      → Prerequisites list
      → Notebook order + descriptions
      → Estimated completion time
      → Link to main README

  ●FILE|id:experiments_readme|location:experiments/README.md
    CONTENT:
      → Purpose: Original research experiments
      → Reference to document/article
      → How to run hallucination_biopsy.py
      → Expected outputs
      → Warning: Requires model download (~5GB)

●DIRECTIVE|phase:configuration_files

  ●FILE|.gitignore
    INCLUDE:
      → Python artifacts (__pycache__, *.pyc, .venv/)
      → Jupyter checkpoints
      → Model files (*.pt, *.safetensors)
      → IDE folders (.vscode/, .cursor/, .idea/)
      → results/*.json (keep structure, ignore data)
      → .DS_Store, Thumbs.db

  ●FILE|pyproject.toml
    CONFIG:
      → name: hallucination-detector
      → version: 0.1.0
      → requires-python: >=3.10
      → dependencies: core packages only
      → optional-dependencies[dev]: pytest, black, flake8

●EXECUTION_ORDER
  1→ Create directory structure
  2→ Write configuration files (.gitignore, pyproject.toml, LICENSE)
  3→ Extract functions to src/hallucination_detector/sae_utils.py
  4→ Copy experiments/hallucination_biopsy.py with updated imports
  5→ Create tutorials/01_sae_basics.ipynb
  6→ Create tutorials/02_feature_extraction.ipynb
  7→ Write README.md files (main, tutorials/, experiments/)
  8→ Verify: pip install -e . works
  9→ Verify: Both notebooks run end-to-end

●VALIDATION_CHECKS
  ✓ All imports resolve correctly
  ✓ No code duplication between src/ and experiments/
  ✓ Notebooks execute without errors
  ✓ README links are not broken
  ✓ Git repo initializes cleanly
  ✓ Existing experiment code still works after migration

●CONSTRAINTS
  ⊕ PRESERVE: All existing code logic, docstrings, variable names
  ⊕ EXTRACT_ONLY: Do not write new implementations
  ⊕ MINIMAL_MODIFICATION: Only change imports and function names where specified
  ≠ NO_GENERATION: Do not create new algorithms or analysis code
  ≠ NO_COMPLEXITY: Keep notebooks simple and educational

●OUTPUT_ARTIFACTS
  Δ → Clean repository structure
  Δ → Two working educational notebooks  
  Δ → Reusable src/ package
  Δ → Preserved experimental code in experiments/
  Δ → Professional README and documentation

●NEXT|after_completion
  → User runs: git init && git add . && git commit -m "Initial repo structure"
  → User tests: jupyter notebook tutorials/01_sae_basics.ipynb
  → User validates: python experiments/hallucination_biopsy.py still works
  → User reviews: README.md for accuracy
  → Ready for Article 1 publication