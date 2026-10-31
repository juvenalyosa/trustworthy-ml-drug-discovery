# Trustworthy Machine Learning for Drug Discovery

### Predicting kinase inhibitor potency — and knowing when to believe your model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juvenalyosa/trustworthy-ml-drug-discovery/blob/main/Trustworthy_ML_for_Drug_Discovery.ipynb)

A model that reports R² = 0.95 and a model that reports R² = 0.60 can describe the same data.
One of them is lying to you. This is a self-contained course — a Colab notebook plus the
full research pipeline behind it — that teaches how to tell the difference.

Everything runs on real measurements: **18,145 curated IC50 values** for inhibitors of
**EGFR**, **ALK** and **BRAF**, including the mutated variants that emerge in patients and
stop responding to treatment.

---

## Start here

Click the badge above. The notebook downloads the curated dataset by itself and runs
end to end on a free Colab CPU — no setup, no account beyond Google, nothing to install.
If the download is unavailable it falls back to a simulated dataset built to reproduce
every lesson, so the notebook never breaks in front of a class.

The notebook is 57 cells and 15 figures across 15 parts, ending with a **workshop**
(6 exercises) and an **exam** (40 points, with answer key).

| Part | Subject |
|---|---|
| Orientation | What machine learning is in one page: the loss function, how a molecule becomes a row of numbers, the four models we fit and how each works |
| 1–2 | When ML is the right tool; pIC50 and the thermodynamics behind the log scale |
| 3–4 | Curating assay data; computing the performance ceiling *before* training |
| 5–6 | Molecular fingerprints and Tanimoto similarity; data splitting and leakage |
| 7–9 | Baselines; gradient boosting derived from its loss function; which metrics to report |
| 10–12 | Negative controls; ranking and virtual screening; a reusable 12-point protocol |
| 13–14 | Workshop and exam |

---

## What the pipeline found

These are the results the notebook teaches you to produce and to distrust.

| Result | Value | Why it matters |
|---|---|---|
| **Honest R²** (scaffold split) | **0.636** | RMSE 0.80 log units; 81% of predictions within 10× |
| Leakage from a random split | **+0.12 R²** | The single most consequential methodological choice |
| Experimental noise ceiling | **σ ≈ 0.92** between publications | The model sits *at* the noise floor — it cannot do much better |
| Docking scores as features | **ΔR² = −0.001** | Useless here; they correlate with molecular *size* (ρ = 0.44), not potency |
| Mutation encoding | ρ = 0.76 on WT→mutant shift | The model does use mutations, but under-predicts their magnitude |
| Protein sequence / ESM2 embeddings | **no gain** over one-hot | 112 models tested; see below |
| Unseen variant penalty | +0.16 RMSE | Generalizing to a *new* mutation is the hard open problem |

### Two negative results worth more than the positive one

**Docking is not predictive here.** Three independent tests agree: Spearman ρ ≤ 0.19 against
measured potency, no improvement when added as a feature, and worse than simple descriptors
on its own. Its apparent signal is molecular size.

**ESM2 protein embeddings did not beat a 22-column one-hot encoding of the mutations.**
Tested with leave-one-variant-out over 16 variants and four embedding strategies. The
one apparent win was an artifact of BRAF V600E carrying 47% of the sample weight; excluding
it, one-hot wins on every uncontaminated metric. A diagnostic settles why: strengthening
the mutation signal made predictions *worse*, so the model is identifying variants, not
understanding them. With only 27 mutant variants in the data there is nothing for a protein
language model to generalize over — it would need kinome-scale data.

Both results were registered as predictions *before* running the experiments. Both held.

---

## Repository contents

The layout is flat on purpose: every script reads its inputs from its own directory, so
the pipeline reproduces without editing a single path.

**Notebook**
```
Trustworthy_ML_for_Drug_Discovery.ipynb   the course
build_notebook.py                         regenerates it (nbformat, never hand-edited JSON)
```

**Pipeline** — run in this order
```
clean_dataset.py            curation: 25,197 raw rows -> 18,145 usable measurements
featurize.py                Morgan/MACCS/Avalon fingerprints -> features_ligand.npz
train_models.py             baselines under 5-fold scaffold CV
tune_models.py              hyperparameter search -> best_params.json
test_sequence_features.py   does the raw sequence help? (leave-one-variant-out)
embed_esm2.py               ESM2 embeddings of the 30 variants
test_esm2.py                do the embeddings help? (same protocol, comparable RMSE)
rank_models.py              regression-as-ranker vs classifier for virtual screening
```

**Data**
```
egfr_alk_braf_merged With ALL docking scores, Cleaned.xlsx   source table
kinase_ml_dataset.csv        curated dataset, 18,145 x 60 — what the notebook loads
variant_sequences.csv        30 variant sequences
features_ligand.npz          precomputed ligand features
esm2_variant_embeddings.npz  ESM2 embeddings (30 x 640)
```

**Results**
```
cleaning_log.txt             the curation funnel, step by step
model_results.csv            baseline comparison
tuning_results.csv           hyperparameter search
best_params.json             the final model
sequence_test_results.csv    sequence encodings, leave-one-variant-out
esm2_test_results.csv        ESM2 encodings, same protocol
oof_predictions*.csv         out-of-fold predictions
```

`fingerprints_ecfp4.csv` (55 MB) is not tracked — `clean_dataset.py` regenerates it.

---

## Reproducing it locally

```bash
conda create -n egfr-ml -c conda-forge python=3.12 \
    rdkit scikit-learn pandas numpy scipy lightgbm matplotlib nbformat
conda activate egfr-ml

python clean_dataset.py     # -> kinase_ml_dataset.csv (byte-identical on rerun)
python featurize.py
python tune_models.py
```

LightGBM needs `libomp`; install it through conda rather than pip or it will fail to load.

The ESM2 step needs its own environment, because `fair-esm` pins an older stack:

```bash
conda create -n esm2 -c conda-forge python=3.11 "numpy<2" "pandas<3" pytorch=2.2.2
conda activate esm2 && pip install fair-esm
python embed_esm2.py
```

On macOS/Intel, `embed_esm2.py` documents three traps worth reading before you debug
them yourself: torch 2.2.2 is the last x86_64 build, importing numpy before torch
segfaults, and `fair-esm`'s hub loader has to be bypassed with
`load_model_and_alphabet_local`.

---

## Data source and licensing

The measurements come from **ChEMBL**, aggregated across 1,371 publications (1989–2024).
ChEMBL data is distributed under **CC BY-SA 3.0**, and the curated derivatives here stay
under that license. The code is MIT (see `LICENSE`).

If you use this in teaching or research, please cite ChEMBL as the underlying data source.
